import cv2
import os
import time
import numpy as np
from typing import List, Tuple

def save_image(image, prefix="capture", folder="capture"):
    """Lưu ảnh vào thư mục 'capture' nằm cùng cấp với file utils.py/main."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{prefix}_{int(time.time())}.jpg"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, image)
    print(f"💾 Saved image: {filepath}")
    return filepath

def draw_center_cross(image, color=(0, 0, 255), size=10, thickness=2, center=None):
    """
    Vẽ dấu + tại center nếu truyền (cx,cy), ngược lại tại chính giữa ảnh.
    Trả về image (được sửa inplace).
    """
    h, w = image.shape[:2]
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = int(round(center[0])), int(round(center[1]))

    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    x1 = max(0, cx - size); x2 = min(w - 1, cx + size)
    y1 = max(0, cy - size); y2 = min(h - 1, cy + size)

    cv2.line(image, (x1, cy), (x2, cy), color, thickness)
    cv2.line(image, (cx, y1), (cx, y2), color, thickness)
    return image

def show_score_popup(
    img: np.ndarray,
    shot_pt: Tuple[int, int],
    center_pt: Tuple[int, int],
    info_texts: List[str],
    window_title: str = "Ket qua"
):
    """
    Vẽ dấu + đỏ tại shot_pt, chấm xanh tại center_pt trên img.
    Hiển thị popup với ảnh + text info.
    """
    img_show = img.copy()
    size = max(8, min(img_show.shape[1], img_show.shape[0]) // 20)
    thickness = 2
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    
    cv2.line(img_show, (shot_pt[0] - size, shot_pt[1]), (shot_pt[0] + size, shot_pt[1]), color_red, thickness)
    cv2.line(img_show, (shot_pt[0], shot_pt[1] - size), (shot_pt[0], shot_pt[1] + size), color_red, thickness)

    radius = size // 2
    cv2.circle(img_show, center_pt, radius, color_blue, thickness=-1)

    h, w = img_show.shape[:2]
    text_area_height = 100
    canvas = np.zeros((h + text_area_height, w, 3), dtype=np.uint8) + 30
    canvas[:h, :, :] = img_show

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness_text = 2
    line_height = 30
    y0 = h + 30

    for i, text in enumerate(info_texts):
        y = y0 + i * line_height
        cv2.putText(canvas, text, (10, y), font, font_scale, font_color, thickness_text, cv2.LINE_AA)

    cv2.imshow(window_title, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)

def show_simple_message(message: str, window_title: str = "Ket qua"):
    img = np.zeros((150, 500, 3), dtype=np.uint8)
    cv2.putText(img, message, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)