# utils/handlers.py

import cv2
import base64
from utils.audio import play_score_sound
from utils.processing import warp_crop_to_original, calculate_score, calculate_score_bia7
from utils.image import save_debug_images, save_training_image

def handle_hit_bia_so_4(hit_info, capture_time, original_img_bia4, mask_bia4):
    """
    Hàm xử lý logic riêng cho bia số 4.
    Trả về một dictionary chứa dữ liệu kết quả để gửi về server.
    """
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']
    
    warped_img, transformed_point = warp_crop_to_original(original_img_bia4, obj_crop, shot_point)
    
    score = 0
    processed_image = original_img_bia4.copy() # Bắt đầu với ảnh bia gốc

    if warped_img is not None and transformed_point is not None:
        print("Đã warp thành công. Đang tính điểm.")
        score = calculate_score(transformed_point, original_img_bia4, mask_bia4)
        cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                       (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3)
    else:
        print("❌ Warp thất bại.")
        # Nếu warp thất bại, vẫn tính điểm trên ảnh crop để có kết quả tương đối
        h_orig, w_orig = original_img_bia4.shape[:2]
        h_crop, w_crop = obj_crop.shape[:2]
        scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
        scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
        scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)
        score = calculate_score(scaled_shot_point, original_img_bia4, mask_bia4)
        cv2.drawMarker(processed_image, scaled_shot_point, 
                       (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu vàng báo lỗi warp
    
    play_score_sound(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)

    return {
        'time': capture_time, 'target': 'Bia số 4', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }

def handle_hit_bia_so_7_8(hit_info, capture_time, original_frame, original_img_bia7, mask_bia7):
    """
    Hàm xử lý logic riêng cho bia số 7-8, bao gồm cả tính điểm.
    """
    save_training_image(original_frame)
    print("✅ Bắn trúng mục tiêu Bia số 7-8. Đang thực hiện warp...")
    
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']
    warped_img, transformed_point = warp_crop_to_original(original_img_bia7, obj_crop, shot_point)
    
    score = 0
    processed_image = original_img_bia7.copy()
    
    if warped_img is not None and transformed_point is not None:
        print("✅ Warp bia 7-8 thành công. Đang tính điểm...")
        # <<< SỬA ĐỔI: Truyền thêm original_img_bia7 và mask_bia7 vào hàm tính điểm >>>
        score = calculate_score_bia7(transformed_point, original_img_bia7, mask_bia7)
        cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                       (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3)
    else:
        print("❌ Warp bia 7-8 thất bại. Đang tính điểm trên ảnh crop.")
        h_orig, w_orig = original_img_bia7.shape[:2]
        h_crop, w_crop = obj_crop.shape[:2]
        
        scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
        scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
        scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)
        
        # <<< SỬA ĐỔI: Truyền thêm original_img_bia7 và mask_bia7 vào hàm tính điểm >>>
        score = calculate_score_bia7(scaled_shot_point, original_img_bia7, mask_bia7)
        cv2.drawMarker(processed_image, scaled_shot_point, 
                       (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3)

    #play_score_sound(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)
    
    return {
        'time': capture_time, 'target': 'Bia số 7-8', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }
def handle_miss(hit_info, capture_time, original_frame):
    """
    Hàm xử lý khi bắn trượt hoặc không phát hiện được.
    Trả về một dictionary chứa dữ liệu kết quả để gửi về server.
    """
    status_text = "Không trúng mục tiêu"
    if hit_info is None:
        status_text = "Không xử lý được"
        print("⚠ Không xử lý được kết quả.")
    else:
        print("❌ Bắn không trúng mục tiêu.")

    play_score_sound(0)
    
    shot_point = hit_info['shot_point']
    processed_image = original_frame.copy()
    cv2.drawMarker(processed_image, shot_point, (0, 0, 255), cv2.MARKER_CROSS, markerSize=30, thickness=2)
    
    _, img_buffer = cv2.imencode('.jpg', processed_image)
    return {
        'time': capture_time, 'target': status_text, 'score': 0,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }