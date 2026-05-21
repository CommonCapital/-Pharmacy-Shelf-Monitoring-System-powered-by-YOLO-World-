from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

model_id = "IDEA-Research/grounding-dino-base"
local_dir = "./local_gdino_model"

print(f"Downloading model {model_id}...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

print(f"Saving to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)
processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print("Done!")
