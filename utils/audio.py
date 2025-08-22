# utils/audio.py - Giải pháp 1: Tăng bộ đệm

import pygame
import os
import time

# --- Khởi tạo mixer với cơ chế chờ đợi và BỘ ĐỆM LỚN HƠN ---
def initialize_mixer():
    """
    Cố gắng khởi tạo pygame mixer. Nếu thất bại, chờ và thử lại.
    """
    while not pygame.mixer.get_init():
        print("⏳ Đang chờ thiết bị âm thanh sẵn sàng...")
        try:
            # <<< THÊM MỚI TẠI ĐÂY >>>
            # Tăng buffer lên 4096 (gấp đôi mặc định) để cho loa có thời gian xử lý
            pygame.mixer.pre_init(44100, -16, 2, 4096) 
            pygame.mixer.init()
        except pygame.error as e:
            print(f"Lỗi tạm thời, sẽ thử lại: {e}")
            time.sleep(2)
    print("✅ Pygame mixer đã khởi tạo thành công!")

initialize_mixer()

# --- Tải trước tất cả âm thanh vào bộ nhớ ---

# Ánh xạ các sự kiện với tên file âm thanh .wav
SCORE_SOUNDS_PATHS = {
    10: "10.wav",
    9: "9.wav",
    8: "8.wav",
    7: "7.wav",
    6: "6.wav",
    5: "5.wav",
    0: "outTarget.wav",
    -1: "connected.wav", # File này được đổi tên từ start.mp3 để nhất quán
    -2: "connected.wav",
    -3: "shot.wav"
}

# Dictionary để lưu các đối tượng âm thanh đã được tải vào RAM
LOADED_SOUNDS = {}

def load_all_sounds():
    """
    Tải tất cả các file âm thanh từ đĩa vào một dictionary trong RAM.
    """
    print("⏳ Đang tải trước các file âm thanh vào bộ nhớ...")
    for code, filename in SCORE_SOUNDS_PATHS.items():
        file_path = os.path.join('sounds', filename)
        if os.path.exists(file_path):
            try:
                LOADED_SOUNDS[code] = pygame.mixer.Sound(file_path)
            except pygame.error as e:
                print(f"Lỗi khi tải file {file_path}: {e}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file âm thanh để tải trước: {file_path}")
    print("✅ Đã tải xong âm thanh!")

load_all_sounds()

# --- Các hàm phát âm thanh (giữ nguyên logic của bạn) ---

def play_sound_from_code(sound_code):
    """
    Phát một âm thanh đã được tải trước từ RAM.
    """
    sound_object = LOADED_SOUNDS.get(sound_code)
    if sound_object:
        try:
            sound_object.play()
        except Exception as e:
            print(f"Lỗi khi phát âm thanh cho mã {sound_code}: {e}")
    else:
        print(f"⚠️ Không tìm thấy âm thanh đã được tải cho mã: {sound_code}")

def play_event_sound(event_type):
    """
    Phát âm thanh cho các sự kiện cụ thể.
    """
    play_sound_from_code(event_type)

def play_score_sound(score):
    """
    Phát âm thanh tương ứng với điểm số.
    """
    if score in LOADED_SOUNDS:
        play_sound_from_code(score)
    else:
        # Nếu điểm không có trong từ điển, mặc định phát âm thanh "bắn trượt"
        play_sound_from_code(0)