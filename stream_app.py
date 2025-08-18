# stream_app.py
from flask import Flask, Response
import threading
import cv2
import queue

# Đổi tên biến Flask
stream_app_instance = Flask(__name__) 
# Sử dụng queue để truyền frame từ luồng chính
frame_queue = queue.Queue(maxsize=1) 

def update_frame(frame):
    """
    Hàm này nhận frame từ luồng chính và cập nhật cho server livestream.
    """
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()  # Xóa frame cũ nếu queue đã đầy
        except queue.Empty:
            pass
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        frame_queue.put(buffer.tobytes())

def generate_frames():
    """
    Hàm tạo luồng frame cho trình duyệt.
    """
    while True:
        try:
            outputFrame = frame_queue.get(timeout=1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + outputFrame + b'\r\n')
            frame_queue.task_done()
        except queue.Empty:
            # Gửi một frame rỗng để tránh lỗi trên trình duyệt
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            continue

@stream_app_instance.route('/video_feed')
def video_feed():
    """
    Endpoint cung cấp luồng video.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')