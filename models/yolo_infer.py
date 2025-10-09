"""yolo_infer.py
script minimal d'exemple pour faire de l'inference YOLO (Ultralytics)
"""
from ultralytics import YOLO
import cv2

def run_video_inference(video_path: str, model_path: str = None):
    model = YOLO(model_path or 'yolov8n.pt')  # adjust if you have local weights
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ultralytics can accept numpy frames
        results = model(frame, verbose=False)
        # results[0].boxes.show()  # placeholder
        print(results[0].boxes.xyxy)  # bbox coords
    cap.release()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python yolo_infer.py path/to/video.mp4')
    else:
        run_video_inference(sys.argv[1])
