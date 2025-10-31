from threading import Thread
import queue
import time
import requests
import evdev
import logging
from evdev import ecodes
from datetime import datetime

from utils.audio import audio_manager

# =================================================================
# [MỚI] Luồng gửi cấu hình đến Server
# =================================================================
class ConfigReporter(Thread):
    def __init__(self, server_url, config_data, shared_state):
        super().__init__()
        self.setName('ConfigReporter')
        self.server_url = server_url
        self.shared_state = shared_state
        self.daemon = True
        self.running = True
        self.session = requests.Session()
        self.reported = False

        # Ưu tiên lấy cấu hình từ 'saved_settings' trước
        saved_settings = config_data.get('saved_settings', {})
        camera_settings = config_data.get('camera', {})

        self.config_to_report = {
            'zoom': saved_settings.get('zoom', camera_settings.get('zoom', 1.0)),
            'center': saved_settings.get('center', camera_settings.get('center', None))
        }

    def run(self):
        logging.info("ConfigReporter bắt đầu hoạt động.")
        time.sleep(1) # Chờ một chút để các luồng khác ổn định
        while self.running:
            try:
                if self.shared_state.server_is_connected and not self.reported:
                    logging.info(f"Đang gửi cấu hình lên server: {self.config_to_report}")
                    self.session.post(
                        f"{self.server_url}/report_config",
                        json=self.config_to_report,
                        timeout=3
                    )
                    self.reported = True
                    logging.info("✅ Gửi cấu hình thành công!")

                elif not self.shared_state.server_is_connected:
                    if self.reported:
                        logging.info("Mất kết nối server, sẵn sàng gửi lại cấu hình khi có kết nối.")
                    self.reported = False

                time.sleep(5)
            except requests.exceptions.RequestException as e:
                logging.warning(f"CONFIG_REPORTER: Lỗi khi gửi cấu hình: {e}")
                self.reported = False
                time.sleep(5)
            except Exception as e:
                logging.error(f"CONFIG_REPORTER: Lỗi không xác định: {e}")
                time.sleep(10)

    def stop(self):
        self.running = False
        if self.session:
            self.session.close()
        logging.info("ConfigReporter đã dừng.")

# =================================================================

class SenderWorker(Thread):
    def __init__(self, frame_queue, server_url, shared_state):
        super().__init__()
        self.setName('SenderWorker')
        self.frame_queue = frame_queue
        self.server_url = server_url
        self.shared_state = shared_state
        self.daemon = True
        self.running = True
        # [TỐI ƯU] Tạo một session duy nhất để tái sử dụng kết nối
        self.session = requests.Session()
        logging.info("SenderWorker đã khởi tạo với requests.Session.")

    def run(self):
        while self.running:
            try:
                if not self.shared_state.server_is_connected:
                    time.sleep(1)
                    continue

                jpg_buffer = self.frame_queue.get(timeout=1)
                try:
                    # [TỐI ƯU] Sử dụng self.session.post thay vì requests.post
                    # Timeout được điều chỉnh hợp lý hơn cho việc gửi dữ liệu
                    self.session.post(
                        f"{self.server_url}/video_upload",
                        data=jpg_buffer,
                        headers={'Content-Type': 'image/jpeg'},
                        timeout=2 # Đặt timeout tổng là 2 giây
                    )
                except requests.exceptions.RequestException as e:
                    if self.shared_state.server_is_connected:
                        # Log lỗi cụ thể hơn để dễ gỡ rối
                        logging.warning(f"SENDER: Mất kết nối khi gửi video. Lỗi: {e}")
                        self.shared_state.server_is_connected = False
                finally:
                    self.frame_queue.task_done()
            except queue.Empty:
                continue
                
    def stop(self):
        self.running = False
        self.session.close() # [TỐI ƯU] Đóng session khi luồng kết thúc
        logging.info("SenderWorker đã dừng và đóng session.")

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
        # [TỐI ƯU] Sử dụng session riêng cho việc hỏi lệnh
        self.session = requests.Session()

    def run(self):
        while self.running:
            try:
                # [TỐI ƯU] Sử dụng self.session.get
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
            
            # [TỐI ƯU] Giảm thời gian chờ xuống để phản ứng nhanh hơn với việc mất kết nối
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
                        logging.debug("Không tìm thấy trigger, đang tìm kiếm lại sau 5 giây...")
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