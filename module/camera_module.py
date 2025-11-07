from picamera2 import Picamera2
import logging

class Camera:
    # THAY ĐỔI 1: Thêm 'shared_state=None' vào hàm khởi tạo
    def __init__(self, width=1280, height=720, shared_state=None):
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(preview_config)
        
        # THAY ĐỔI 2: Lưu shared_state làm thuộc tính (để luồng stream truy cập)
        self.shared_state = shared_state
        
        # THAY ĐỔI 3: Thêm cờ trạng thái
        self._is_running = False
        logging.info("Module camera đã được khởi tạo.")

#khởi động
    def start(self):
        self.picam2.start()
        # THAY ĐỔI 4: Đặt cờ là True khi camera đã chạy
        self._is_running = True
        logging.info("Camera đã bắt đầu.")

#chụp hình
    def capture_frame(self):
        if not self._is_running:
            return None
        return self.picam2.capture_array()

#thoát
    def stop(self):
        if self._is_running:
            self.picam2.stop()
            # THAY ĐỔI 5: Đặt cờ là False khi camera dừng
            self._is_running = False
            logging.info("Camera đã dừng.")

    # THAY ĐỔI 6: Thêm hàm is_running() mà app.py (luồng stream) cần
    def is_running(self):
        return self._is_running