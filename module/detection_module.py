# module/detection_module.py

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="my_model.pt"):
        """
        Khởi tạo detector với model YOLO.
        """
        try:
            self.model = YOLO(model_path)
            # In ra thông tin các lớp mà model có thể nhận diện
            print(f"✅ Model YOLO đã được tải thành công. Các lớp: {self.model.names}")
        except Exception as e:
            print(f"❌ Lỗi khi tải model YOLO: {e}")
            self.model = None

    def detect(self, image, conf=0.5):
        """
        Thực hiện nhận dạng đối tượng trên ảnh.

        Args:
            image: Ảnh đầu vào (định dạng OpenCV).
            conf: Ngưỡng tin cậy.

        Returns:
            Một danh sách các dictionary, mỗi dictionary chứa thông tin về một vật thể được phát hiện.
            Ví dụ: [{'box': [x1, y1, x2, y2], 'conf': 0.95, 'class_name': 'bia_so_4'}]
        """
        if self.model is None:
            return []

        detections = []
        results = self.model(image, conf=conf, verbose=False) # verbose=False để log gọn hơn
        
        if results and results[0].boxes:
            res = results[0]
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            class_ids = res.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes_xyxy, confs, class_ids):
                class_name = self.model.names[int(cls_id)]
                detections.append({
                    'box': [int(coord) for coord in box],
                    'conf': float(conf),
                    'class_name': class_name
                })
        
        return detections