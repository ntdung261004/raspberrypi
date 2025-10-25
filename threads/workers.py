from threading import Thread
import queue
import time
import requests
import evdev
import logging
from evdev import ecodes
from datetime import datetime

from utils.audio import audio_manager

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