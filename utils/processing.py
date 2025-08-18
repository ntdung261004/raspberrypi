import cv2
import numpy as np
from typing import Optional, Tuple, List
import os

def friendly_object_name(filename: str) -> str:
    base = filename.split('/')[-1]
    name, _ = base.split('.') if '.' in base else (base, '')
    return name.replace('_', ' ')

def check_object_center(results, image, conf_threshold=0.5):
    h, w = image.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    if not results or not results[0].boxes:
        print("⚠ Không tìm thấy object.")
        return "TRƯỢT", None, None

    res = results[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    for box, conf in zip(boxes_xyxy, confs):
        if conf < conf_threshold:
            continue
        
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        if x1 <= cx_img <= x2 and y1 <= cy_img <= y2:
            print(f"✅ TRÚNG | Confidence: {conf:.2f}")

            orig_w = x2 - x1
            orig_h = y2 - y1
            if orig_w <= 0 or orig_h <= 0:
                continue

            obj_crop = image[y1:y2, x1:x2].copy()
            obj_crop = cv2.resize(obj_crop, (500, 500), interpolation=cv2.INTER_AREA)

            scale_x = 500.0 / orig_w
            scale_y = 500.0 / orig_h
            cx_crop = int(round((cx_img - x1) * scale_x))
            cy_crop = int(round((cy_img - y1) * scale_y))

            return "TRÚNG", obj_crop, (cx_crop, cy_crop)

    print("❌ TRƯỢT")
    return "TRƯỢT", None, None

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
    good12 = [m for m, n in matches12 if m.distance < ratio_thresh * n.distance]
    good21 = [m for m, n in matches21 if m.distance < ratio_thresh * n.distance]

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

def calculate_score(pt: Tuple[float, float], original_img: np.ndarray, mask: np.ndarray) -> int:
    if original_img is None or mask is None:
        return 0
    x, y = int(pt[0]), int(pt[1])
    h, w = original_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    center_x, center_y = w // 2, h // 2
    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    
    if mask[y, x] == 255:
        if distance < 56:
            return 10
        elif distance < 116:
            return 9
        elif distance < 173:
            return 8
        elif distance < 230:
            return 7
        elif distance < 285:
            return 6
        elif distance < 320:
            return 5
    return 0