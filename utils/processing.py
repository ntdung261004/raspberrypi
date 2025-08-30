import cv2
import numpy as np
from typing import Optional, Tuple, List
import os

def friendly_object_name(filename: str) -> str:
    base = filename.split('/')[-1]
    name, _ = base.split('.') if '.' in base else (base, '')
    return name.replace('_', ' ')

def check_object_center(detections, image, calibrated_center):
    """
    Kiểm tra xem tâm ngắm có nằm trong bounding box của vật thể nào không.

    Args:
        detections: Danh sách các vật thể đã được nhận dạng từ ObjectDetector.
        image: Khung hình gốc.
        calibrated_center: Tọa độ tâm đã hiệu chỉnh (hoặc None).

    Returns:
        Tuple: (status, hit_info)
        status (str): "TRÚNG" hoặc "TRƯỢT".
        hit_info (dict): Thông tin của vật thể bị trúng, hoặc tọa độ tâm ngắm nếu trượt.
    """
    # Xác định tâm ngắm sẽ sử dụng
    if calibrated_center:
        center_x, center_y = calibrated_center['x'], calibrated_center['y']
    else:
        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2

    # Tìm vật thể bị bắn trúng có độ tin cậy cao nhất
    highest_conf_hit = None
    for det in detections:
        x1, y1, x2, y2 = det['box']
        
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            if highest_conf_hit is None or det['conf'] > highest_conf_hit['conf']:
                highest_conf_hit = det

    if highest_conf_hit:
        x1, y1, x2, y2 = highest_conf_hit['box']
        obj_crop = image[y1:y2, x1:x2].copy()
        shot_point_relative = (center_x - x1, center_y - y1)
        
        hit_info = {
            'name': highest_conf_hit['class_name'],
            'crop': obj_crop,
            'shot_point': shot_point_relative,
            'conf': highest_conf_hit['conf']
        }
        print(f"✅ TRÚNG | Mục tiêu: {hit_info['name']} (Conf: {hit_info['conf']:.2f})")
        return "TRÚNG", hit_info
    
    print("❌ TRƯỢT | Tâm ngắm không nằm trong bất kỳ mục tiêu nào.")
    return "TRƯỢT", {'shot_point': (center_x, center_y)}

def warp_crop_to_original(
    original_img: np.ndarray,
    obj_crop: np.ndarray,
    shot_point: Optional[Tuple[float, float]] = None,
    min_inliers: int = 10,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 4.0,
    max_reproj: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    if original_img is None or obj_crop is None:
        print("[warp_crop_to_original] ERROR: Ảnh đầu vào bị None")
        return None, None

    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    kp1, des1 = orb.detectAndCompute(original_img, None)
    kp2, des2 = orb.detectAndCompute(obj_crop, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[warp_crop_to_original] Không đủ đặc trưng để match.")
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches12 = bf.knnMatch(des1, des2, k=2)
    matches21 = bf.knnMatch(des2, des1, k=2)
    
    # Lọc các điểm match tốt bằng Lowe's ratio test
    good12 = [m for m, n in matches12 if m.distance < ratio_thresh * n.distance]
    good21 = [m for m, n in matches21 if m.distance < ratio_thresh * n.distance]

    # Lọc các điểm match tương hỗ (mutual matches)
    mutual = []
    reverse_map = {(m.trainIdx, m.queryIdx) for m in good21}
    for m in good12:
        if (m.queryIdx, m.trainIdx) in reverse_map:
            mutual.append(m)

    if len(mutual) < min_inliers:
        print(f"[warp_crop_to_original] Mutual matches quá ít: {len(mutual)}")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
    if H is None or abs(np.linalg.det(H)) < 1e-6:
        print("[warp_crop_to_original] Homography không hợp lệ hoặc suy biến.")
        return None, None

    transformed_point = None
    if shot_point is not None:
        try:
            px, py = float(shot_point[0]), float(shot_point[1])
            src_pt = np.array([[[px, py]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(src_pt, H)[0][0]
            transformed_point = (float(warped_pt[0]), float(warped_pt[1]))
            print(f"[warp_crop_to_original] Tọa độ vết đạn chuyển sang ảnh gốc: {transformed_point}")
        except Exception as e:
            print(f"[warp_crop_to_original] Lỗi chuyển tọa độ điểm: {e}")

    print("[warp_crop_to_original] Warp ảnh thành công")
    warped = cv2.warpPerspective(obj_crop, H, (original_img.shape[1], original_img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, transformed_point

#tính điểm bia 8
def calculate_score_bia8(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    """
    Tính điểm cho bia số 8 dựa trên các vòng elip.
    """
    if pt is None or mask is None or original_img is None:
        return 0

    x, y = int(pt[0]), int(pt[1])
    # Lấy kích thước từ ảnh gốc thay vì ảnh mask để đảm bảo chính xác
    h, w = original_img.shape[:2]

    if not (0 <= y < h and 0 <= x < w) or mask[y, x] == 0:
        return 0

    center_x, center_y = 87, 116

    ellipse_rings = [
        {'score': 10, 'width': 42,  'height': 63},
        {'score': 9,  'width': 84, 'height': 126},
        {'score': 8,  'width': 126, 'height': 190},
        {'score': 7,  'width': 172, 'height': 258},
        {'score': 6,  'width': 216, 'height': 324},
        {'score': 5,  'width': 260, 'height': 324},
        {'score': 4,  'width': 304, 'height': 456},
        {'score': 3,  'width': 348, 'height': 522},
        {'score': 2,  'width': 392, 'height': 588},
        {'score': 1,  'width': 436, 'height': 654}
    ]

    for ring in ellipse_rings:
        a = ring['width'] / 2.0
        b = ring['height'] / 2.0
        if a > 0 and b > 0:
            check = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2)
            if check <= 1:
                return ring['score']
    
    return 0

#tính điểm bia 7
def calculate_score_bia7(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    """
    Tính điểm cho bia số 7 dựa trên các vòng elip.
    """
    if pt is None or mask is None or original_img is None:
        return 0

    x, y = int(pt[0]), int(pt[1])
    # Lấy kích thước từ ảnh gốc thay vì ảnh mask để đảm bảo chính xác
    h, w = original_img.shape[:2]

    if not (0 <= y < h and 0 <= x < w) or mask[y, x] == 0:
        return 0

    center_x, center_y = 136, 177

    ellipse_rings = [
        {'score': 10, 'width': 63,  'height': 95},
        {'score': 9,  'width': 126, 'height': 190},
        {'score': 8,  'width': 189, 'height': 284},
        {'score': 7,  'width': 252, 'height': 378},
        {'score': 6,  'width': 309, 'height': 464},
        {'score': 5,  'width': 375, 'height': 562},
        {'score': 4,  'width': 436, 'height': 654},
        {'score': 3,  'width': 497, 'height': 746},
        {'score': 2,  'width': 557, 'height': 836},
        {'score': 1,  'width': 613, 'height': 920}
    ]

    for ring in ellipse_rings:
        a = ring['width'] / 2.0
        b = ring['height'] / 2.0
        if a > 0 and b > 0:
            check = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2)
            if check <= 1:
                return ring['score']
    
    return 0

#tính điểm bia số 4
def calculate_score_bia4(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    if original_img is None or mask is None or pt is None:
        return 0
    x, y = int(pt[0]), int(pt[1])
    h, w = original_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    center_x, center_y = w // 2, h // 2
    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    
    # Kiểm tra xem điểm chạm có nằm trong vùng hợp lệ của bia không
    if mask[y, x] == 255:
        if distance < 56: return 10
        elif distance < 116: return 9
        elif distance < 173: return 8
        elif distance < 230: return 7
        elif distance < 285: return 6
        elif distance < 320: return 5
    
    return 0