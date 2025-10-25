# threads/workers.py

from threading import Thread
import queue
import time
import requests
import evdev
import logging
from evdev import ecodes
from datetime import datetime
import base64

# <<< BẮT ĐẦU PHẦN THÊM MỚI >>>
# 1. Import thư viện socketio client
import socketio
# <<< KẾT THÚC PHẦN THÊM MỚI >>>

from utils.audio import audio_manager

# =======================================================================
# === SENDERWORKER ĐƯỢC VIẾT LẠI HOÀN TOÀN ĐỂ DÙNG WEBSOCKET ===
# =======================================================================

class SenderWorker(Thread):
    def __init__(self, frame_queue, server_url, shared_state):
        super().__init__()
        self.setName('SenderWorker')
        self.frame_queue = frame_queue
        self.server_url = server_url
        self.shared_state = shared_state
        self.daemon = True
        self.running = True
        
        # 2. Khởi tạo Socket.IO client thay vì requests.Session
        self.sio = socketio.Client(reconnection_delay_max=5, logger=False, engineio_logger=False)
        self.is_connected = False

        # --- Định nghĩa các hàm xử lý sự kiện cho client ---
        @self.sio.event
        def connect():
            logging.info("✅ SENDER: Đã kết nối thành công tới server qua WebSocket!")
            self.is_connected = True
            self.shared_state.server_is_connected = True

        @self.sio.event
        def connect_error(data):
            # Chỉ log lỗi khi đang ở trạng thái kết nối để tránh spam log
            if self.is_connected:
                logging.error(f"❌ SENDER: Lỗi kết nối WebSocket: {data}")
            self.is_connected = False
            self.shared_state.server_is_connected = False

        @self.sio.event
        def disconnect():
            logging.warning("⚠️ SENDER: Đã mất kết nối WebSocket tới server. Đang thử kết nối lại...")
            self.is_connected = False
            self.shared_state.server_is_connected = False

    def _connect_to_server(self):
        """Vòng lặp chạy nền để cố gắng kết nối đến server."""
        while self.running and not self.is_connected:
            try:
                logging.info(f"SENDER: Đang thử kết nối WebSocket tới {self.server_url}...")
                self.sio.connect(self.server_url, transports=['websocket'])
                # Đợi một chút để kết nối được thiết lập hoàn toàn
                self.sio.sleep(1)
            except Exception as e:
                logging.warning(f"SENDER: Không thể kết nối, thử lại sau 3 giây... Lỗi: {e}")
                self.sio.sleep(3)

    def run(self):
        self._connect_to_server() # Bắt đầu quá trình kết nối
        
        last_heartbeat_time = 0
        while self.running:
            if not self.is_connected:
                # Nếu mất kết nối, client sẽ tự động thử kết nối lại.
                # Luồng này chỉ cần đợi.
                self.sio.sleep(1)
                continue

            try:
                # Lấy khung hình đã nén JPEG từ hàng đợi
                jpg_buffer = self.frame_queue.get(timeout=1)
                
                # 3. Mã hóa dữ liệu ảnh sang Base64 để gửi qua WebSocket
                b64_string = base64.b64encode(jpg_buffer).decode('utf-8')
                
                # 4. Gửi khung hình qua sự kiện 'pi_stream'
                self.sio.emit('pi_stream', {'image': b64_string})

                # 5. Gửi "nhịp tim" (heartbeat) mỗi 2 giây để server biết Pi vẫn còn sống
                current_time = time.time()
                if current_time - last_heartbeat_time > 2:
                    self.sio.emit('pi_heartbeat', {'timestamp': current_time})
                    last_heartbeat_time = current_time

            except queue.Empty:
                # Hàng đợi rỗng là chuyện bình thường, không cần log lỗi
                continue
            except Exception as e:
                logging.error(f"SENDER: Lỗi trong vòng lặp chính: {e}")
                self.sio.sleep(1)

        # Dọn dẹp khi luồng dừng lại
        if self.is_connected:
            self.sio.disconnect()
        logging.info("SenderWorker đã dừng.")

    def stop(self):
        self.running = False

# =======================================================================
# === CÁC LỚP WORKER KHÁC ĐƯỢC GIỮ NGUYÊN, KHÔNG THAY ĐỔI ===
# =======================================================================

class CommandPoller(Thread):
    def __init__(self, command_queue, server_url, shared_state):
        super().__init__()
        self.setName('CommandPoller')
        self.command_queue = command_queue
        self.server_url = server_url
        self.shared_state = shared_state
        self.daemon = True
        self.running = True
        self.last_server_heartbeat = 0
        self.session = requests.Session()

    def run(self):
        while self.running:
            try:
                response = self.session.get(f"{self.server_url}/get_command", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('timestamp'): 
                        if not self.shared_state.server_is_connected:
                            logging.info("✅ POLLER: Khôi phục kết nối tới server!")
                        self.shared_state.server_is_connected = True
                        self.last_server_heartbeat = time.time()
                    
                    command = data.get('command')
                    if command and not self.command_queue.full():
                        self.command_queue.put(command)

            except requests.exceptions.RequestException:
                if time.time() - self.last_server_heartbeat > 5:
                    if self.shared_state.server_is_connected:
                        logging.warning("POLLER: Mất kết nối tới server.")
                        self.shared_state.server_is_connected = False
            
            time.sleep(2) 
            
    def stop(self):
        self.running = False
        self.session.close()
        logging.info("CommandPoller đã dừng và đóng session.")

class TriggerListener(Thread):
    def __init__(self, processing_queue, ring_buffer, config, shared_state):
        super().__init__()
        self.setName('TriggerListener')
        self.processing_queue = processing_queue
        self.ring_buffer = ring_buffer
        self.shared_state = shared_state
        self.daemon = True
        self.running = True
        self.device = None
        self.device_name_keyword = config['trigger']['device_keyword']

    def find_trigger_device(self):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            if self.device_name_keyword.lower() in device.name.lower():
                logging.info(f"Đã tìm thấy thiết bị trigger: {device.name} tại {device.path}")
                return device
        return None

    def run(self):
        logging.info("Luồng TriggerListener bắt đầu hoạt động...")
        while self.running:
            try:
                if self.device is None:
                    self.device = self.find_trigger_device()
                    if self.device is None:
                        logging.info("Không tìm thấy trigger, đang tìm kiếm lại sau 5 giây...")
                        time.sleep(5)
                        continue
                    else:
                         self.device.grab()
                         logging.info(f"Đã giành quyền kiểm soát '{self.device.name}'. Bắt đầu lắng nghe...")

                for event in self.device.read_loop():
                    if not self.running: break
                    
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_VOLUMEUP and event.value == 1:
                        audio_manager.play_shot() 
                        capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        logging.info(f"📸 (BT Trigger) Nhận tín hiệu lúc {capture_time}")

                        if len(self.ring_buffer) > 0 and not self.processing_queue.full():
                            frame_to_process = self.ring_buffer[0]
                            self.processing_queue.put((frame_to_process.copy(), capture_time))
                        else:
                            logging.warning("Ring buffer rỗng hoặc processing queue đầy, bỏ qua trigger.")

            except (IOError, OSError) as e:
                logging.warning(f"Thiết bị trigger đã bị ngắt kết nối: {e}. Đang tìm kiếm lại...")
                if self.device:
                    try: self.device.close()
                    except: pass
                self.device = None
                time.sleep(2)
        
        if self.device:
            try:
                self.device.ungrab()
                self.device.close()
            except: pass
        logging.info("TriggerListener đã dừng.")

    def stop(self):
        self.running = False