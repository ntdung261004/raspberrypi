from picamera2 import Picamera2

class Camera:
    def __init__(self, width=1280, height=720):
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(preview_config)

#khởi động
    def start(self):
        self.picam2.start()
#chụp hình
    def capture_frame(self):
        return self.picam2.capture_array()
#thoát
    def stop(self):
        self.picam2.stop()
