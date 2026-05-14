from ultralytics import YOLOWorld
import numpy as np

model = YOLOWorld("my_yolo_world_model.pt")
model.set_classes(["NuroFen", "box", "medicine box", "white and red box"])

img = np.zeros((640, 640, 3), dtype=np.uint8)
img[100:200, 100:200] = 255 # create a dummy white box

results = model.predict(img, conf=0.005)
for r in results:
    print("Classes:", r.names)
    print("Detected classes:", r.boxes.cls.tolist())
    print("Detected confs:", r.boxes.conf.tolist())
