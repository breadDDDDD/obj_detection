from ultralytics import YOLO

models_to_test = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
    'yolov9n.pt', 'yolov10n.pt', 'yolo11n.pt', 'yolov11n.pt'
]

print("Testing available YOLO models:")
for model in models_to_test:
    try:
        YOLO(model)
        print(f" {model} - Available")
    except:
        print(f" {model} - Not available")