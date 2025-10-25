import cv2
import queue
import requests
import logging
import os
from threading import Thread

# Import các module chức năng
from utils.processing import check_object_center
from utils.handles import handle_hit_bia_so_4b, handle_hit_bia_so_4c, handle_miss

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
            self.original_img_bia4b = cv2.imread(paths['original_bia4b'])
            self.mask_bia4b = cv2.imread(paths['mask_bia4b'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia4b_alt = cv2.imread(paths['original_bia4b_alt'])
            # ... Tương tự cho các bia khác
            self.original_img_bia4c = cv2.imread(paths['original_bia4c'])
            self.mask_bia4c = cv2.imread(paths['mask_bia4c'], cv2.IMREAD_GRAYSCALE)
            self.original_img_bia4c_alt = cv2.imread(paths['original_bia4c_alt'])
            # Kiểm tra nếu bất kỳ file nào không tải được
            if self.original_img_bia4b is None or self.mask_bia4b is None:
                raise FileNotFoundError("Một hoặc nhiều file tài nguyên cho bia 4b không tồn tại.")
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
            if target_name == 'bia_4b':
                result_data = handle_hit_bia_so_4b(hit_info, capture_time, frame, self.original_img_bia4b, self.original_img_bia4b_alt, self.mask_bia4b)
            elif target_name == 'bia_4c':
                result_data = handle_hit_bia_so_4c(hit_info, capture_time, frame, self.original_img_bia4c, self.original_img_bia4c_alt, self.mask_bia4c)
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