from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="my_model.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔍 Detector running on: {device}")
        self.model = YOLO(model_path).to(device)
        self.device = device

    def detect(self, frame, conf=0.5):
        return self.model.predict(frame, conf=conf, device=self.device, verbose=False)
 # nhận diện mặt bia tin cậy trên 50%