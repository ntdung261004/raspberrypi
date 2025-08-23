# utils/images.py

import os
import cv2
from datetime import datetime

def save_debug_images(original_frame, yolo_crop, warped_result=None):
    """
    Lưu một bộ ảnh debug vào một thư mục con được đặt tên theo timestamp.
    (Hàm này vẫn được giữ lại)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join("debug_images", timestamp)
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, "1_frame_goc.jpg"), original_frame)
        cv2.imwrite(os.path.join(output_dir, "2_yolo_crop.jpg"), yolo_crop)
        
        if warped_result is not None:
            cv2.imwrite(os.path.join(output_dir, "3_warped_result.jpg"), warped_result)
        
        print(f"✅ Đã lưu ảnh debug vào thư mục: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu ảnh debug: {e}")
        return None

# <<< THÊM MỚI: Hàm để lưu ảnh cho việc training >>>
def save_training_image(frame):
    """
    Lưu một khung hình gốc vào thư mục data_image với tên file duy nhất.
    """
    output_dir = "data_image"
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo tên file duy nhất bằng timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        file_path = os.path.join(output_dir, filename)
        
        # Lưu ảnh
        cv2.imwrite(file_path, frame)
        print(f"🖼️  Đã lưu ảnh training: {file_path}")
        return file_path
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu ảnh training: {e}")
        return None