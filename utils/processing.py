import cv2
import numpy as np
from typing import Optional, Tuple, List
import os
import logging

logger = logging.getLogger(__name__)

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

def _find_bounding_rect_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Tìm contour lớn nhất và trả về 4 góc của hình chữ nhật bao quanh nó.
    Thứ tự các góc luôn cố định, đảm bảo không bị lật ngược.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Trả về 4 góc theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    return corners

def warp_via_bounding_rect(original_img: np.ndarray, cropped_img: np.ndarray, shot_point_relative: Tuple[float, float]) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """
    Ánh xạ ảnh crop vào ảnh bia gốc bằng hình chữ nhật bao quanh.
    Đây là phương pháp ổn định và chống lật ngược hình ảnh.
    """
    try:
        logger.info("Bắt đầu warp bằng phương pháp Bounding Rectangle...")
        dst_pts = _find_bounding_rect_corners(original_img)
        src_pts = _find_bounding_rect_corners(cropped_img)

        if dst_pts is None or src_pts is None:
            logger.warning("Warp thất bại: Không thể tìm thấy bounding rect trên ảnh nguồn hoặc ảnh đích.")
            return None, None
            
        # Sử dụng getPerspectiveTransform cho 4 điểm, chính xác và nhanh hơn
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        if M is None:
            logger.warning("Warp thất bại: getPerspectiveTransform không thể tính toán ma trận.")
            return None, None
            
        shot_point_np = np.float32([[shot_point_relative]]).reshape(-1, 1, 2)
        transformed_point_np = cv2.perspectiveTransform(shot_point_np, M)
        
        transformed_point = (transformed_point_np[0][0][0], transformed_point_np[0][0][1])
        logger.info("Warp bằng Bounding Rectangle thành công.")
        return M, transformed_point

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong hàm warp_via_bounding_rect: {e}", exc_info=True)
        return None, None


#tính điểm bia số 4
def calculate_score_bia4b(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    if original_img is None or mask is None or pt is None:
        return 0
    x, y = int(pt[0]), int(pt[1])
    h, w = original_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    center_x, center_y = 254, 250
    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    
    if mask[y, x] == 255:
        if distance < 23: return 10
        elif distance < 45: return 9
        elif distance < 69: return 8
        elif distance < 92: return 7
        elif distance < 115: return 6
        elif distance < 137: return 5
        elif distance < 162: return 4
        elif distance < 185: return 3
        elif distance < 208: return 2
        elif distance < 231: return 1
    return 0

def calculate_score_bia4c(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    if original_img is None or mask is None or pt is None:
        return 0
    x, y = int(pt[0]), int(pt[1])
    h, w = original_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    center_x, center_y = 249, 248.5
    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    
    if mask[y, x] == 255:
        if distance < 46: return 10
        elif distance < 83: return 9
        elif distance < 121: return 8
        elif distance < 158: return 7
        elif distance < 194: return 6
        elif distance < 231: return 5
    return 0