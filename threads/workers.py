# threads/workers.py

from threading import Thread
import queue
import time
import requests
import evdev
from evdev import ecodes
from datetime import datetime

# Import các biến và hàm cần thiết từ các module khác
from utils.audio import play_event_sound
# <<< THÊM MỚI: Import biến CALIBRATED_CENTER từ main.py >>>
# Đây là một cách để luồng phụ có thể truy cập biến của luồng chính
import __main__ as main_module

class SenderWorker(Thread):
    def __init__(self, frame_queue, server_url):
        super().__init__()
        self.frame_queue = frame_queue
        self.server_url = server_url
        self.daemon = True
        self.running = True

    def run(self):
        while self.running:
            try:
                if not main_module.SERVER_IS_CONNECTED:
                    time.sleep(1)
                    continue

                jpg_buffer = self.frame_queue.get_nowait()
                try:
                    requests.post(f"{self.server_url}/video_upload", data=jpg_buffer, headers={'Content-Type': 'image/jpeg'}, timeout=(2, 5))
                except requests.exceptions.RequestException as e:
                    # <<< THAY ĐỔI: Chủ động báo mất kết nối >>>
                    if main_module.SERVER_IS_CONNECTED:
                        print("‼️ LỖI SENDER: Mất kết nối khi gửi video. Tạm dừng.")
                        main_module.SERVER_IS_CONNECTED = False
                finally:
                    self.frame_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
                continue
                
    def stop(self):
        self.running = False

class CommandPoller(Thread):
    def __init__(self, command_queue, server_url):
        super().__init__()
        self.command_queue = command_queue
        self.server_url = server_url
        self.daemon = True
        self.running = True
        self.last_server_heartbeat = 0

    def run(self):
        while self.running:
            try:
                response = requests.get(f"{self.server_url}/get_command", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('timestamp'):
                        if not main_module.SERVER_IS_CONNECTED:
                            print("✅ Khôi phục kết nối tới server!")
                        main_module.SERVER_IS_CONNECTED = True
                        self.last_server_heartbeat = time.time()
                    
                    command = data.get('command')
                    if command:
                        self.command_queue.put(command)

            except requests.exceptions.RequestException:
                # <<< THAY ĐỔI: Chủ động báo mất kết nối nếu cần >>>
                if time.time() - self.last_server_heartbeat > 5: # Rút ngắn thời gian chờ xuống 5s
                    if main_module.SERVER_IS_CONNECTED:
                        print("‼️ LỖI POLLER: Mất kết nối. Tạm dừng.")
                        main_module.SERVER_IS_CONNECTED = False
            
            time.sleep(2) # Tăng thời gian hỏi lệnh để giảm tải mạng
            
    def stop(self):
        self.running = False

class TriggerListener(Thread):
    def __init__(self, processing_queue, ring_buffer):
        super().__init__()
        self.processing_queue = processing_queue
        self.ring_buffer = ring_buffer
        self.daemon = True
        self.running = True
        self.device = None
        self.device_name_keyword = "AB Shutter3" # Thay bằng tên remote của bạn

    def find_trigger_device(self):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            if self.device_name_keyword.lower() in device.name.lower():
                print(f"✅ Đã tìm thấy thiết bị trigger: {device.name} tại {device.path}")
                return device
        return None

    def run(self):
        print("🎧 Luồng TriggerListener bắt đầu hoạt động...")
        while self.running:
            try:
                if self.device is None:
                    self.device = self.find_trigger_device()
                    if self.device is None:
                        print(f"🔎 Không tìm thấy trigger, đang tìm kiếm lại sau 5 giây...")
                        time.sleep(5)
                        continue
                    else:
                         self.device.grab()
                         print(f"✅ Giành quyền kiểm soát {self.device.name}. Bắt đầu lắng nghe...")

                for event in self.device.read():
                    if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_VOLUMEDOWN and event.value == 1:
                        capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"📸 (BT Trigger) Chụp ảnh lúc {capture_time}...")
                        play_event_sound(-3)

                        # Truy cập biến CALIBRATED_CENTER được truyền vào từ main
                        if len(self.ring_buffer) > 0 and not self.processing_queue.full():
                            frame_to_process = self.ring_buffer[0]
                            # Cần truy cập CALIBRATED_CENTER, sẽ được xử lý trong main.py
                            self.processing_queue.put((frame_to_process.copy(), capture_time, main_module.CALIBRATED_CENTER))

            except BlockingIOError:
                time.sleep(0.05)
                continue
            except (IOError, OSError) as e:
                print(f"⚠️ Thiết bị trigger đã bị ngắt kết nối: {e}. Đang tìm kiếm lại...")
                if self.device:
                    try: self.device.close()
                    except: pass
                self.device = None
                time.sleep(2)

    def stop(self):
        print("🔌 Yêu cầu dừng TriggerListener...")
        self.running = False