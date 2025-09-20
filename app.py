import cv2
import queue
import requests
import logging
import os
from threading import Thread

# Import các module chức năng
from utils.processing import check_object_center
from utils.handles import handle_hit_bia_so_4, handle_hit_bia_so_7, handle_hit_bia_so_8, handle_miss

class ProcessingWorker(Thread):
    # <<< TỐI ƯU: Nhận config và shared_state từ main >>>
    def __init__(self, process_queue, detector, server_url, config, shared_state):
        super().__init__()
        self.setName('ProcessingWorker') # Đặt tên cho luồng để log rõ ràng hơn
        self.process_queue = process_queue
        self.detector = detector
        self.server_url = server_url
        self.config = config
        self.shared_state = shared_state # Đối tượng trạng thái chia sẻ
        
        self._load_resources()

        self.daemon = True
        self.running = True
        logging.info("ProcessingWorker đã khởi động.")

    def _load_resources(self):
        """Tải các tài nguyên ảnh và mask từ đường dẫn trong config."""
        logging.info("Đang tải tài nguyên hình ảnh cho các loại bia...")
        paths = self.config['resource_paths']
        
        # <<< TỐI ƯU: Thêm try-except để kiểm tra file tồn tại >>>
        try:
            self.original_img_bia4 = cv2.imread(paths['original_bia4'])
            self.mask_bia4 = cv2.imread(paths['mask_bia4'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia4_alt = cv2.imread(paths['original_bia4_alt'])
            # ... Tương tự cho các bia khác
            self.original_img_bia7 = cv2.imread(paths['original_bia7'])
            self.mask_bia7 = cv2.imread(paths['mask_bia7'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia7_alt = cv2.imread(paths['original_bia7_alt'])
            self.original_img_bia8 = cv2.imread(paths['original_bia8'])
            self.mask_bia8 = cv2.imread(paths['mask_bia8'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia8_alt = cv2.imread(paths['original_bia8_alt'])
            
            # Kiểm tra nếu bất kỳ file nào không tải được
            if self.original_img_bia4 is None or self.mask_bia4 is None:
                raise FileNotFoundError("Một hoặc nhiều file tài nguyên cho bia 4 không tồn tại.")
        except Exception as e:
            logging.error(f"Lỗi nghiêm trọng khi tải tài nguyên: {e}. Vui lòng kiểm tra đường dẫn trong config.json.")
            # Có thể thoát chương trình ở đây nếu tài nguyên là bắt buộc
            os._exit(1)
    
    def run(self):
        while self.running:
            try:
                frame, capture_time = self.process_queue.get(timeout=1)
                # <<< TỐI ƯU: Lấy center_coords từ shared_state >>>
                center_coords = self.shared_state.calibrated_center
                self._process_frame(frame, capture_time, center_coords)
                self.process_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Lỗi không xác định trong luồng xử lý: {e}", exc_info=True)
    
    def _send_results(self, result_data):
        """Hàm riêng để gửi kết quả về server."""
        try:
            requests.post(f"{self.server_url}/processed_data_upload", json=result_data, timeout=5)
            logging.info("🚀 Gửi dữ liệu xử lý lên server thành công.")
        except requests.exceptions.RequestException as e:
            # <<< TỐI ƯU: Sử dụng shared_state để quản lý kết nối >>>
            if self.shared_state.server_is_connected:
                logging.warning(f"Mất kết nối khi gửi dữ liệu xử lý: {e}")
                self.shared_state.server_is_connected = False

    def _process_frame(self, frame, capture_time, center_coords):
        """Hàm điều phối, gọi đến các handler tương ứng."""
        detections = self.detector.detect(frame, conf=self.config['model']['confidence'])
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