from ultralytics import YOLO
import numpy as np

model = YOLO("my_yolo_world_model.pt")
model.set_classes(["NuroFen", "box", "medicine"])

# Create a dummy image
img = np.zeros((640, 640, 3), dtype=np.uint8)
# Draw a white box to simulate something
img[100:200, 100:200] = 255

results = model.predict(img, conf=0.005)
for r in results:
    print("Detected classes:", r.boxes.cls.tolist())
    print("Detected confs:", r.boxes.conf.tolist())

