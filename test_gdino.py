import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

img = np.zeros((640, 640, 3), dtype=np.uint8)
img[100:200, 100:200] = 255 # create a dummy white box

text = "white box. black background."

inputs = processor(images=img, text=text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.2,
    text_threshold=0.2,
    target_sizes=[img.shape[:2]]
)[0]

print("Boxes:", results["boxes"])
print("Labels:", results["labels"])
print("Scores:", results["scores"])
