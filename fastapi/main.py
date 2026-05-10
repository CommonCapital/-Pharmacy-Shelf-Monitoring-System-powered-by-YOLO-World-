from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLOWorld
import supervision as sv
import json
import os
from typing import List

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="YOLO-World Prototype Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
print("Loading YOLO-World model...")
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_yolo_world_model.pt")
model = YOLOWorld(model_path)

@app.post("/process-video")
async def process_video(video: UploadFile = File(...), config: str = Form(...)):
    """
    Takes a video and a JSON config (ground truth boxes).
    Runs YOLO-World on the video and compares against the config.
    """
    ground_truth = json.loads(config) # List of {x1, y1, x2, y2, label}
    classes = [item['label'] for item in ground_truth]
    model.set_classes(classes)

    # Save temporary video file
    video_path = f"temp_{video.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(video_path)
    final_results = []
    
    # Process only a few frames for the prototype demonstration
    success, frame = cap.read()
    if success:
        results = model.predict(frame)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Compare each ground truth box to live detections
        for gt in ground_truth:
            # Find detection in the GT region
            found = "Empty"
            is_match = False
            
            for i in range(len(detections.xyxy)):
                det_box = detections.xyxy[i]
                det_label = classes[detections.class_id[i]]
                
                # Simple center point check
                cx = (det_box[0] + det_box[2]) / 2
                cy = (det_box[1] + det_box[3]) / 2
                
                if cx >= gt['x1'] and cx <= gt['x2'] and cy >= gt['y1'] and cy <= gt['y2']:
                    found = det_label
                    if det_label == gt['label']:
                        is_match = True
                    break
            
            final_results.append({
                "label": gt['label'],
                "detected": found,
                "match": is_match
            })

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)

    return final_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
