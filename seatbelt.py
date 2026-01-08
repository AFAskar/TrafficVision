from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")
coco_yaml = "./Seat_belt_detection-2/data.yaml"

results = model.train(data=coco_yaml, epochs=5, imgsz=640, device=device)

# save the model
model.export(weights="yolo11n-seatbelt.pt")
