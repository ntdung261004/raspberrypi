# utils/handlers.py

import cv2
import base64
from utils.audio import play_score_sound
from utils.processing import warp_crop_to_original, calculate_score_bia4, calculate_score_bia7, calculate_score_bia8
from utils.image import save_debug_images, save_training_image

def handle_hit_bia_so_7(hit_info, capture_time, original_frame, 
                        original_img_bia7, original_img_bia7_alt, 
                        mask_bia7):
    """
    Hàm xử lý logic riêng cho bia số 7.
    Thử warp với 2 ảnh gốc, nếu đều thất bại mới tính điểm trên ảnh crop.
    """
    save_training_image(original_frame)
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']
    
    score = 0
    # Luôn chuẩn bị ảnh gốc đầu tiên để vẽ kết quả cuối cùng lên đó
    processed_image = original_img_bia7.copy() 
    
    # === BƯỚC 1: Thử warp với ảnh gốc thứ nhất ===
    print("✅ Bắn trúng bia 7. Thử warp với ảnh gốc 1...")
    warped_img, transformed_point = warp_crop_to_original(original_img_bia7, obj_crop, shot_point)
    
    if warped_img is not None and transformed_point is not None:
        print("✅ Warp (ảnh 1) thành công. Đang tính điểm...")
        score = calculate_score_bia7(transformed_point, original_img_bia7, mask_bia7)
        cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                       (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu đỏ: thành công ở lần 1
    else:
        # === BƯỚC 2: Thử warp với ảnh gốc thứ hai nếu BƯỚC 1 thất bại ===
        print("❌ Warp (ảnh 1) thất bại. Thử warp với ảnh gốc 2...")
        warped_img_alt, transformed_point_alt = warp_crop_to_original(original_img_bia7_alt, obj_crop, shot_point)
        
        if warped_img_alt is not None and transformed_point_alt is not None:
            print("✅ Warp (ảnh 2) thành công. Đang tính điểm...")
            # Tính điểm với dữ liệu của ảnh thứ hai
            score = calculate_score_bia7(transformed_point_alt, original_img_bia7_alt, mask_bia7)
            # Vẽ điểm đã biến đổi lên ảnh gốc đầu tiên để có kết quả nhất quán
            cv2.drawMarker(processed_image, (int(transformed_point_alt[0]), int(transformed_point_alt[1])), 
                           (0, 165, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu cam: thành công ở lần 2
        else:
            # === BƯỚC 3: Dùng ảnh crop nếu cả hai lần warp đều thất bại ===
            print("❌ Warp (ảnh 2) cũng thất bại. Tính điểm trên ảnh crop.")
            h_orig, w_orig = original_img_bia7.shape[:2]
            h_crop, w_crop = obj_crop.shape[:2]
            
            scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
            scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
            scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)
            
            score = calculate_score_bia7(scaled_shot_point, original_img_bia7, mask_bia7)
            cv2.drawMarker(processed_image, scaled_shot_point, 
                           (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu vàng: warp thất bại

    play_score_sound(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)
    
    return {
        'time': capture_time, 'target': 'Bia số 7', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }
  
def handle_hit_bia_so_8(hit_info, capture_time, original_frame, 
                        original_img_bia8, original_img_bia8_alt, 
                        mask_bia8):
    """
    Hàm xử lý logic riêng cho bia số 8.
    Thử warp với 2 ảnh gốc, nếu đều thất bại mới tính điểm trên ảnh crop.
    """
    save_training_image(original_frame)
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']
    
    score = 0
    # Luôn chuẩn bị ảnh gốc đầu tiên để vẽ kết quả cuối cùng lên đó
    processed_image = original_img_bia8.copy() 
    
    # === BƯỚC 1: Thử warp với ảnh gốc thứ nhất ===
    print("✅ Bắn trúng bia 8. Thử warp với ảnh gốc 1...")
    warped_img, transformed_point = warp_crop_to_original(original_img_bia8, obj_crop, shot_point)
    
    if warped_img is not None and transformed_point is not None:
        print("✅ Warp (ảnh 1) thành công. Đang tính điểm...")
        score = calculate_score_bia8(transformed_point, original_img_bia8, mask_bia8)
        cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                       (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu đỏ: thành công ở lần 1
    else:
        # === BƯỚC 2: Thử warp với ảnh gốc thứ hai nếu BƯỚC 1 thất bại ===
        print("❌ Warp (ảnh 1) thất bại. Thử warp với ảnh gốc 2...")
        warped_img_alt, transformed_point_alt = warp_crop_to_original(original_img_bia8_alt, obj_crop, shot_point)
        
        if warped_img_alt is not None and transformed_point_alt is not None:
            print("✅ Warp (ảnh 2) thành công. Đang tính điểm...")
            # Tính điểm với dữ liệu của ảnh thứ hai
            score = calculate_score_bia7(transformed_point_alt, original_img_bia8_alt, mask_bia8)
            # Vẽ điểm đã biến đổi lên ảnh gốc đầu tiên để có kết quả nhất quán
            cv2.drawMarker(processed_image, (int(transformed_point_alt[0]), int(transformed_point_alt[1])), 
                           (0, 165, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu cam: thành công ở lần 2
        else:
            # === BƯỚC 3: Dùng ảnh crop nếu cả hai lần warp đều thất bại ===
            print("❌ Warp (ảnh 2) cũng thất bại. Tính điểm trên ảnh crop.")
            h_orig, w_orig = original_img_bia8.shape[:2]
            h_crop, w_crop = obj_crop.shape[:2]
            
            scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
            scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
            scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)
            
            score = calculate_score_bia8(scaled_shot_point, original_img_bia8, mask_bia8)
            cv2.drawMarker(processed_image, scaled_shot_point, 
                           (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu vàng: warp thất bại

    play_score_sound(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)
    
    return {
        'time': capture_time, 'target': 'Bia số 8', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }
  
def handle_hit_bia_so_4(hit_info, capture_time, original_frame, 
                        original_img_bia4, original_img_bia4_alt, 
                        mask_bia4):
    """
    Hàm xử lý logic riêng cho bia số 4.
    Thử warp với 2 ảnh gốc, nếu đều thất bại mới tính điểm trên ảnh crop.
    """
    save_training_image(original_frame)
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']
    
    score = 0
    # Luôn chuẩn bị ảnh gốc đầu tiên để vẽ kết quả cuối cùng lên đó
    processed_image = original_img_bia4.copy() 
    
    # === BƯỚC 1: Thử warp với ảnh gốc thứ nhất ===
    print("✅ Bắn trúng bia 4. Thử warp với ảnh gốc 1...")
    warped_img, transformed_point = warp_crop_to_original(original_img_bia4, obj_crop, shot_point)
    
    if warped_img is not None and transformed_point is not None:
        print("✅ Warp (ảnh 1) thành công. Đang tính điểm...")
        score = calculate_score_bia4(transformed_point, original_img_bia4, mask_bia4)
        cv2.drawMarker(processed_image, (int(transformed_point[0]), int(transformed_point[1])), 
                       (0, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu đỏ: thành công ở lần 1
    else:
        # === BƯỚC 2: Thử warp với ảnh gốc thứ hai nếu BƯỚC 1 thất bại ===
        print("❌ Warp (ảnh 1) thất bại. Thử warp với ảnh gốc 2...")
        warped_img_alt, transformed_point_alt = warp_crop_to_original(original_img_bia4_alt, obj_crop, shot_point)
        
        if warped_img_alt is not None and transformed_point_alt is not None:
            print("✅ Warp (ảnh 2) thành công. Đang tính điểm...")
            # Tính điểm với dữ liệu của ảnh thứ hai
            score = calculate_score_bia4(transformed_point_alt, original_img_bia4_alt, mask_bia4)
            # Vẽ điểm đã biến đổi lên ảnh gốc đầu tiên để có kết quả nhất quán
            cv2.drawMarker(processed_image, (int(transformed_point_alt[0]), int(transformed_point_alt[1])), 
                           (0, 165, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu cam: thành công ở lần 2
        else:
            # === BƯỚC 3: Dùng ảnh crop nếu cả hai lần warp đều thất bại ===
            print("❌ Warp (ảnh 2) cũng thất bại. Tính điểm trên ảnh crop.")
            h_orig, w_orig = original_img_bia4.shape[:2]
            h_crop, w_crop = obj_crop.shape[:2]
            
            scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
            scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
            scaled_shot_point = (scaled_shot_point_x, scaled_shot_point_y)

            score = calculate_score_bia4(scaled_shot_point, original_img_bia4, mask_bia4)
            cv2.drawMarker(processed_image, scaled_shot_point,
                           (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=3) # Màu vàng: warp thất bại

    play_score_sound(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)
    
    return {
        'time': capture_time, 'target': 'Bia số 4', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }
    
def handle_miss(hit_info, capture_time, original_frame):
    """
    Hàm xử lý khi bắn trượt hoặc không phát hiện được.
    Trả về một dictionary chứa dữ liệu kết quả để gửi về server.
    """
    save_training_image(original_frame)
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