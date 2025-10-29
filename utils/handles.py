# utils/handles.py

import cv2
import base64
from utils.audio import audio_manager
from utils.processing import warp_via_bounding_rect, calculate_score_bia4b, calculate_score_bia4c
from utils.image import save_debug_images, save_training_image


def handle_hit_bia_so_4b(hit_info, capture_time, original_frame,
                        original_img_bia4b, original_img_bia4b_alt,
                        mask_bia4b):
    """
    Hàm xử lý logic riêng cho bia số 4b.
    Vẽ vết đạn to hơn và luôn là màu đỏ.
    """
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']

    score = 0
    processed_image = original_img_bia4b.copy()
    
    # [THAY ĐỔI] Quy định màu và kích thước mới
    HIT_COLOR = (0, 0, 255)  # Luôn là màu đỏ
    CIRCLE_RADIUS = 25       # Bán kính vòng tròn to hơn
    MARKER_SIZE = 30         # Kích thước dấu thập to hơn
    THICKNESS = 2            # Độ dày nét vẽ

    print("✅ Bắn trúng bia 4b. Thử warp với ảnh gốc 1...")
    warped_img, transformed_point = warp_via_bounding_rect(original_img_bia4b, obj_crop, shot_point)

    if warped_img is not None and transformed_point is not None:
        print("✅ Warp (ảnh 1) thành công. Đang tính điểm...")
        score = calculate_score_bia4b(transformed_point, original_img_bia4b, mask_bia4b)
        point_to_draw = (int(transformed_point[0]), int(transformed_point[1]))
        
        # Vẽ theo style mới
        cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
        cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)
    else:
        print("❌ Warp (ảnh 1) thất bại. Thử warp với ảnh gốc 2...")
        warped_img_alt, transformed_point_alt = warp_via_bounding_rect(original_img_bia4b_alt, obj_crop, shot_point)

        if warped_img_alt is not None and transformed_point_alt is not None:
            print("✅ Warp (ảnh 2) thành công. Đang tính điểm...")
            score = calculate_score_bia4b(transformed_point_alt, original_img_bia4b_alt, mask_bia4b)
            point_to_draw = (int(transformed_point_alt[0]), int(transformed_point_alt[1]))

            # Vẽ theo style mới
            cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
            cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)
        else:
            print("❌ Warp (ảnh 2) cũng thất bại. Tính điểm trên ảnh crop.")
            h_orig, w_orig = original_img_bia4b.shape[:2]
            h_crop, w_crop = obj_crop.shape[:2]

            scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
            scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
            point_to_draw = (scaled_shot_point_x, scaled_shot_point_y)

            score = calculate_score_bia4b(point_to_draw, original_img_bia4b, mask_bia4b)

            # Vẽ theo style mới
            cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
            cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)

    audio_manager.play_score(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)

    return {
        'time': capture_time, 'target': 'Bia số 4b', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }

def handle_hit_bia_so_4c(hit_info, capture_time, original_frame,
                        original_img_bia4c, original_img_bia4c_alt,
                        mask_bia4c):
    """
    Hàm xử lý logic riêng cho bia số 4c.
    Vẽ vết đạn to hơn và luôn là màu đỏ.
    """
    obj_crop = hit_info['crop']
    shot_point = hit_info['shot_point']

    score = 0
    processed_image = original_img_bia4c.copy()

    # [THAY ĐỔI] Quy định màu và kích thước mới
    HIT_COLOR = (0, 0, 255)  # Luôn là màu đỏ
    CIRCLE_RADIUS = 25       # Bán kính vòng tròn to hơn
    MARKER_SIZE = 30         # Kích thước dấu thập to hơn
    THICKNESS = 2           # Độ dày nét vẽ

    print("✅ Bắn trúng bia 4c. Thử warp với ảnh gốc 1...")
    warped_img, transformed_point = warp_via_bounding_rect(original_img_bia4c, obj_crop, shot_point)

    if warped_img is not None and transformed_point is not None:
        print("✅ Warp (ảnh 1) thành công. Đang tính điểm...")
        score = calculate_score_bia4c(transformed_point, original_img_bia4c, mask_bia4c)
        point_to_draw = (int(transformed_point[0]), int(transformed_point[1]))

        # Vẽ theo style mới
        cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
        cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)
    else:
        print("❌ Warp (ảnh 1) thất bại. Thử warp với ảnh gốc 2...")
        warped_img_alt, transformed_point_alt = warp_via_bounding_rect(original_img_bia4c_alt, obj_crop, shot_point)

        if warped_img_alt is not None and transformed_point_alt is not None:
            print("✅ Warp (ảnh 2) thành công. Đang tính điểm...")
            score = calculate_score_bia4c(transformed_point_alt, original_img_bia4c_alt, mask_bia4c)
            point_to_draw = (int(transformed_point_alt[0]), int(transformed_point_alt[1]))

            # Vẽ theo style mới
            cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
            cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)
        else:
            print("❌ Warp (ảnh 2) cũng thất bại. Tính điểm trên ảnh crop.")
            h_orig, w_orig = original_img_bia4c.shape[:2]
            h_crop, w_crop = obj_crop.shape[:2]

            scaled_shot_point_x = int(shot_point[0] * w_orig / w_crop)
            scaled_shot_point_y = int(shot_point[1] * h_orig / h_crop)
            point_to_draw = (scaled_shot_point_x, scaled_shot_point_y)

            score = calculate_score_bia4c(point_to_draw, original_img_bia4c, mask_bia4c)
            
            # Vẽ theo style mới
            cv2.circle(processed_image, point_to_draw, CIRCLE_RADIUS, (255, 255, 255), THICKNESS)
            cv2.drawMarker(processed_image, point_to_draw, HIT_COLOR, markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=THICKNESS)

    audio_manager.play_score(score)
    _, img_buffer = cv2.imencode('.jpg', processed_image)

    return {
        'time': capture_time, 'target': 'Bia số 4c', 'score': score,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }

def handle_miss(hit_info, capture_time, original_frame):
    """
    Hàm xử lý khi bắn trượt.
    Vẽ vết đạn theo style mới.
    """
    status_text = "Không trúng mục tiêu"
    if hit_info is None:
        status_text = "Không xử lý được"
        print("⚠ Không xử lý được kết quả.")
    else:
        print("❌ Bắn không trúng mục tiêu.")

    audio_manager.play_miss()

    shot_point = hit_info['shot_point']
    processed_image = original_frame.copy()

    # [THAY ĐỔI] Vẽ hồng tâm to hơn và luôn màu đỏ cho trường hợp bắn trượt
    cv2.circle(processed_image, shot_point, 16, (255, 255, 255), 1)
    cv2.drawMarker(processed_image, shot_point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    _, img_buffer = cv2.imencode('.jpg', processed_image)
    return {
        'time': capture_time, 'target': status_text, 'score': 0,
        'image_data': base64.b64encode(img_buffer).decode('utf-8')
    }