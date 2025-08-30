import sys
import cv2
import queue
import requests
from threading import Thread
import base64

# Import các module chức năng
from module.detection_module import ObjectDetector
from utils.processing import check_object_center
from utils.handles import handle_hit_bia_so_4, handle_hit_bia_so_7, handle_hit_bia_so_8, handle_miss

# Đường dẫn tài nguyên
ORIGINAL_IMAGE_BIA4_PATH = "images/original/bia_so_4.png"
ORIGINAL_IMAGE_BIA4_ALT_PATH = "images/original/bia_so_4_1.png"
MASK_BIA4_PATH = "images/mask/mask_bia_so_4.png"
ORIGINAL_IMAGE_BIA7_PATH = "images/original/bia_so_7.png"
ORIGINAL_IMAGE_BIA7_ALT_PATH = "images/original/bia_so_7_1.png"
MASK_BIA7_PATH = "images/mask/mask_bia_so_7.png"
ORIGINAL_IMAGE_BIA8_PATH = "images/original/bia_so_8.png"
ORIGINAL_IMAGE_BIA8_ALT_PATH = "images/original/bia_so_8_1.png"
MASK_BIA8_PATH = "images/mask/mask_bia_so_8.png"

class ProcessingWorker(Thread):
    def __init__(self, process_queue, detector, server_url):
        super().__init__()
        self.process_queue = process_queue
        self.detector = detector
        self.server_url = server_url # <<< NHẬN URL TỪ BÊN NGOÀI
        
        print("💡 Đang tải tài nguyên cho các loại bia...")
        self.original_img_bia4 = cv2.imread(ORIGINAL_IMAGE_BIA4_PATH)
        self.mask_bia4 = cv2.imread(MASK_BIA4_PATH, cv2.IMREAD_GRAYSCALE)
        self.original_img_bia4_alt = cv2.imread(ORIGINAL_IMAGE_BIA4_ALT_PATH)
        
        self.original_img_bia7 = cv2.imread(ORIGINAL_IMAGE_BIA7_PATH)
        self.mask_bia7 = cv2.imread(MASK_BIA7_PATH, cv2.IMREAD_GRAYSCALE)
        self.original_img_bia7_alt = cv2.imread(ORIGINAL_IMAGE_BIA7_ALT_PATH)

        self.original_img_bia8 = cv2.imread(ORIGINAL_IMAGE_BIA8_PATH)
        self.mask_bia8 = cv2.imread(MASK_BIA8_PATH, cv2.IMREAD_GRAYSCALE)
        self.original_img_bia8_alt = cv2.imread(ORIGINAL_IMAGE_BIA8_ALT_PATH)

        self.daemon = True
        self.running = True
        print("💡 ProcessingWorker đã khởi động.")
    
    def run(self):
        while self.running:
            try:
                frame, capture_time, center_coords = self.process_queue.get(timeout=1)
                self._process_frame(frame, capture_time, center_coords)
                self.process_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi trong luồng xử lý: {e}")
    
    def _send_results(self, result_data):
        """Hàm riêng để gửi kết quả về server."""
        try:
            # <<< DÙNG URL ĐÃ ĐƯỢC TRUYỀN VÀO >>>
            requests.post(f"{self.server_url}/processed_data_upload", json=result_data, timeout=5)
            print("🚀 Đã gửi dữ liệu xử lý lên server thành công.")
        except requests.exceptions.RequestException as e:
            # <<< THAY ĐỔI: Chủ động báo mất kết nối >>>
            if main_module.SERVER_IS_CONNECTED:
                print(f"❌ Lỗi khi gửi dữ liệu xử lý. Mất kết nối. {e}")
                main_module.SERVER_IS_CONNECTED = False

    def _process_frame(self, frame, capture_time, center_coords):
        """Hàm điều phối, gọi đến các handler tương ứng."""
        detections = self.detector.detect(frame, conf=0.75)
        status, hit_info = check_object_center(detections, frame, center_coords)
        
        result_data = None
        
        if status == "TRÚNG":
            target_name = hit_info.get('name')
            if target_name == 'bia_so_4':
                result_data = handle_hit_bia_so_4(hit_info, capture_time, frame, self.original_img_bia4, self.original_img_bia4_alt, self.mask_bia4)
            elif target_name == 'bia_so_7_8':
                result_data = handle_hit_bia_so_7(hit_info, capture_time, frame, self.original_img_bia7, self.original_img_bia7_alt, self.mask_bia7)
            elif target_name == 'bia_so_8':
                result_data = handle_hit_bia_so_8(hit_info, capture_time, frame, self.original_img_bia8, self.original_img_bia8_alt, self.mask_bia8)
            else:
                print(f"Phát hiện trúng mục tiêu không xác định: {target_name}")
                result_data = handle_miss(hit_info, capture_time, frame)
        
        else: # TRƯỢT
            result_data = handle_miss(hit_info, capture_time, frame)

        if result_data:
            self._send_results(result_data)

    def stop(self):
        self.running = False
        print("🛑 ProcessingWorker đã dừng.")