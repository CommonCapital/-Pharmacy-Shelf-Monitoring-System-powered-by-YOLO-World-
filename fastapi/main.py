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
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="YOLO-World Prototype Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for saving annotated videos/images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally
print("Loading YOLO-World model...")
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_yolo_world_model.pt")
model = YOLOWorld(model_path)

@app.post("/process-video")
async def process_video(video: UploadFile = File(...), config: str = Form(...)):
    """
    Takes a video/image and a JSON config (ground truth boxes).
    Runs YOLO-World, annotates the media, and compares against the config.
    """
    ground_truth = json.loads(config) # List of {x1, y1, x2, y2, label}
    classes = [item['label'] for item in ground_truth]
    # Keep unique classes for YOLO-World
    unique_classes = list(set(classes))
    if unique_classes:
        model.set_classes(unique_classes)

    # Save temporary file
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    is_image = temp_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    
    detected_json = []
    final_results = []
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    if is_image:
        frame = cv2.imread(temp_path)
        results = model.predict(frame, conf=0.005, imgsz=1024)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Annotate YOLO detections
        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Draw Expected Ground Truth boxes
        for gt in ground_truth:
            x1, y1, x2, y2 = int(gt['x1']), int(gt['y1']), int(gt['x2']), int(gt['y2'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for expected
            cv2.putText(annotated_frame, f"Expected: {gt['label']}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        output_filename = f"out_{video.filename}"
        output_path = os.path.join("static", output_filename)
        cv2.imwrite(output_path, annotated_frame)
        
        # Populate detected_json
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            label = model.names[detections.class_id[i]]
            conf = float(detections.confidence[i])
            detected_json.append({
                "label": label,
                "confidence": conf,
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3])
            })
    else:
        cap = cv2.VideoCapture(temp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"out_{video.filename}"
        if not output_filename.endswith('.mp4'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp4'
        output_path = os.path.join("static", output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        first_frame_processed = False
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = model.predict(frame, verbose=False, conf=0.005, imgsz=1024)
            detections = sv.Detections.from_ultralytics(results[0])
            
            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Draw Expected Ground Truth boxes
            for gt in ground_truth:
                x1, y1, x2, y2 = int(gt['x1']), int(gt['y1']), int(gt['x2']), int(gt['y2'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for expected
                cv2.putText(annotated_frame, f"Expected: {gt['label']}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            out.write(annotated_frame)
            
            # Extract detections from the first frame for comparison
            if not first_frame_processed:
                first_frame_processed = True
                for i in range(len(detections.xyxy)):
                    box = detections.xyxy[i]
                    label = model.names[detections.class_id[i]]
                    conf = float(detections.confidence[i])
                    detected_json.append({
                        "label": label,
                        "confidence": conf,
                        "x1": float(box[0]), "y1": float(box[1]),
                        "x2": float(box[2]), "y2": float(box[3])
                    })
                    
        cap.release()
        out.release()
        
    # Compare
    for gt in ground_truth:
        best_match_label = "Empty"
        highest_conf = -1.0
        is_match = False
        
        for det in detected_json:
            cx = (det['x1'] + det['x2']) / 2
            cy = (det['y1'] + det['y2']) / 2
            
            # Add a 20% margin to the bounding box to handle slight angle/lighting shifts
            w = gt['x2'] - gt['x1']
            h = gt['y2'] - gt['y1']
            margin_x = w * 0.2
            margin_y = h * 0.2
            
            if (gt['x1'] - margin_x) <= cx <= (gt['x2'] + margin_x) and (gt['y1'] - margin_y) <= cy <= (gt['y2'] + margin_y):
                if det['confidence'] > highest_conf:
                    highest_conf = det['confidence']
                    best_match_label = det['label']
                    
        if best_match_label == gt['label']:
            is_match = True
                
        final_results.append({
            "label": gt['label'],
            "detected": best_match_label,
            "match": is_match,
            "expected_box": {"x1": gt['x1'], "y1": gt['y1'], "x2": gt['x2'], "y2": gt['y2']}
        })

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "media_url": f"http://localhost:8000/static/{output_filename}",
        "detected_json": detected_json,
        "comparison": final_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
