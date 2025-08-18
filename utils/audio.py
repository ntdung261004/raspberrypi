import os
from threading import Thread
import subprocess

# Ánh xạ các sự kiện với tên file âm thanh
SCORE_SOUNDS = {
    10: "10.mp3",
    9: "9.mp3",
    8: "8.mp3",
    7: "7.mp3",
    6: "6.mp3",
    5: "5.mp3",
    0: "outTarget.mp3", # Bắn trượt hoặc không xử lý được
    -1: "connected.mp3", # Âm thanh khởi động
    -2: "connected.mp3", # Âm thanh kết nối server thành công
    -3: "shot.mp3" # Âm thanh thông báo đã bắn
}

def play_sound(filename):
    """
    Phát một file âm thanh bằng mpg123 trong một luồng riêng.
    """
    file_path = os.path.join('sounds', filename)
    if not os.path.exists(file_path):
        print(f"File âm thanh không tồn tại: {file_path}")
        return

    def _play():
        try:
            # -q để chạy ở chế độ im lặng, chỉ phát âm thanh
            subprocess.run(['mpg123', '-q', file_path], check=True)
        except FileNotFoundError:
            print("mpg123 không được cài đặt. Vui lòng chạy: sudo apt-get install mpg123")
        except Exception as e:
            print(f"Lỗi khi phát âm thanh {file_path}: {e}")

    audio_thread = Thread(target=_play, daemon=True)
    audio_thread.start()

def play_event_sound(event_type):
    """
    Phát âm thanh cho các sự kiện cụ thể.
    """
    filename = SCORE_SOUNDS.get(event_type)
    if filename:
        play_sound(filename)

def play_score_sound(score):
    """
    Phát âm thanh tương ứng với điểm số.
    """
    if score in SCORE_SOUNDS:
        play_sound(SCORE_SOUNDS[score])
    else:
        # Nếu điểm không có trong từ điển, mặc định phát âm thanh "bắn trượt"
        play_event_sound(0)