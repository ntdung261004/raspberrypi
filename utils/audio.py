# utils/audio.py
import pygame
import os
import logging

# --- Cài đặt logging cơ bản để nhận thông báo ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class AudioManager:
    """
    Lớp quản lý tập trung tất cả các chức năng âm thanh của ứng dụng.
    Sử dụng mẫu thiết kế Singleton (được tạo một lần duy nhất).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AudioManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Biến cờ để đảm bảo __init__ chỉ chạy một lần
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.sounds = {}
        try:
            # Sử dụng cài đặt mặc định của pygame, đã được tối ưu
            pygame.mixer.init()
            logging.info("✅ Pygame mixer đã khởi tạo thành công!")
            self._load_all_sounds()
        except pygame.error as e:
            logging.error(f"❌ Lỗi nghiêm trọng khi khởi tạo pygame.mixer: {e}. Âm thanh sẽ không hoạt động.")
            self.sounds = None # Vô hiệu hóa âm thanh nếu có lỗi

    def _load_all_sounds(self):
        """
        Tải tất cả các file âm thanh vào bộ nhớ (RAM).
        Phương thức này được gọi tự động khi khởi tạo.
        """
        if self.sounds is None:
            return

        logging.info("⏳ Đang tải trước các file âm thanh vào bộ nhớ...")
        
        # Tự động xác định đường dẫn đến thư mục 'sounds'
        # Giả định file audio.py nằm trong thư mục 'utils', và 'sounds' ngang cấp với 'utils'
        sounds_dir = os.path.join(os.path.dirname(__file__), '..', 'sounds')

        # Ánh xạ TÊN LOGIC với TÊN FILE
        sound_map = {
            # Tên sự kiện
            "shot": "shot.mp3",
            "miss": "outTarget.mp3",
            "connected": "connected.mp3",
            # Tên điểm số
            "score_1": "1.mp3",
            "score_2": "2.mp3",
            "score_3": "3.mp3",
            "score_4": "4.mp3",
            "score_5": "5.mp3",
            "score_6": "6.mp3",
            "score_7": "7.mp3",
            "score_8": "8.mp3",
            "score_9": "9.mp3",
            "score_10": "10.mp3"
        }

        for name, filename in sound_map.items():
            file_path = os.path.join(sounds_dir, filename)
            if os.path.exists(file_path):
                try:
                    self.sounds[name] = pygame.mixer.Sound(file_path)
                except pygame.error as e:
                    logging.error(f"⚠️ Lỗi khi tải file {file_path}: {e}")
            else:
                logging.warning(f"🔍 Không tìm thấy file âm thanh: {file_path}")
        
        logging.info("✅ Đã tải xong âm thanh!")

    def play_sound(self, name: str):
        """
        Phát một âm thanh dựa trên TÊN LOGIC của nó.
        Đây là phương thức cốt lõi.
        """
        if self.sounds is None:
            logging.warning("Không thể phát âm thanh vì mixer chưa được khởi tạo.")
            return
        
        sound_object = self.sounds.get(name)
        if sound_object:
            try:
                sound_object.play()
            except Exception as e:
                logging.error(f"Lỗi khi phát âm thanh '{name}': {e}")
        else:
            logging.warning(f"Không tìm thấy âm thanh có tên: '{name}'")

    # --- Các hàm tiện ích (Helper Methods) để gọi dễ hơn ---

    def play_score(self, score: int):
        """Phát âm thanh cho một điểm số cụ thể (từ 1 đến 10)."""
        if 1 <= score <= 10:
            self.play_sound(f"score_{score}")
        else:
            # Nếu điểm không hợp lệ, mặc định phát tiếng bắn trượt
            self.play_miss()
            logging.warning(f"Điểm số không hợp lệ ({score}), phát âm thanh 'miss'.")

    def play_shot(self):
        """Phát âm thanh tiếng súng bắn."""
        print("Thực hiện phát âm thanh!")
        self.play_sound("shot")

    def play_miss(self):
        """Phát âm thanh bắn trượt mục tiêu."""
        self.play_sound("miss")

    def play_connected(self):
        """Phát âm thanh kết nối thành công."""
        self.play_sound("connected")

# --- TẠO RA MỘT THỂ HIỆN (INSTANCE) DUY NHẤT CỦA AUDIOMANAGER ---
# Đây là chìa khóa để sử dụng dễ dàng trong toàn bộ dự án.
audio_manager = AudioManager()