from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import supervision as sv
import json
import os
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Grounding DINO Prototype Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading Grounding DINO model (Base)...")
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
gdino_detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
gdino_detector.to(device)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    box format: [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

@app.post("/process-video")
async def process_video(video: UploadFile = File(...), config: str = Form(...)):
    try:
        ground_truth = json.loads(config)
    except Exception as e:
        return {"error": f"Invalid config JSON: {str(e)}"}
        
    classes = [item['label'] for item in ground_truth]
    unique_classes = list(set(classes))
    
    text_prompt = ""
    if unique_classes:
        # Grounding DINO works better with spaces around dots
        text_prompt = " . ".join(unique_classes).lower() + " ."

    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    is_image = temp_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.jfif'))
    
    detected_json = []
    final_results = []
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def process_frame_targeted(frame, gt_boxes):
        """
        Processes the frame by cropping into each GT box and running DINO on the crop.
        """
        h, w = frame.shape[:2]
        print(f"DEBUG: Processing frame of size {w}x{h}")
        
        slot_results = []
        all_detections_for_viz = []
        
        # We can batch these for efficiency if needed, but let's start with a loop
        for gt in gt_boxes:
            x1_gt, y1_gt, x2_gt, y2_gt = gt['x1'], gt['y1'], gt['x2'], gt['y2']
            
            # Add 20% padding to the crop to give DINO some context
            pad_w = (x2_gt - x1_gt) * 0.2
            pad_h = (y2_gt - y1_gt) * 0.2
            
            crop_x1 = max(0, int(x1_gt - pad_w))
            crop_y1 = max(0, int(y1_gt - pad_h))
            crop_x2 = min(w, int(x2_gt + pad_w))
            crop_y2 = min(h, int(y2_gt + pad_h))
            
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                slot_results.append({"label": gt['label'], "detected": "Empty", "match": False, "conf": 0, "expected_box": gt})
                continue
            
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # --- PASS 1: Competitive Check (All brands) ---
            # This identifies if another brand is a better match to prevent "Agreement Bias"
            prompt_comp = " . ".join(unique_classes).lower() + " ."
            inputs_c = processor(images=rgb_crop, text=prompt_comp, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_c = gdino_detector(**inputs_c)
            results_c = processor.post_process_grounded_object_detection(
                outputs_c, inputs_c.input_ids, threshold=0.15, text_threshold=0.2, target_sizes=[crop.shape[:2]]
            )[0]

            # --- PASS 2: Targeted Check (Expected brand ONLY) ---
            # This gives the expected brand (like Fanta) the full attention of the model
            prompt_target = f"{gt['label'].lower()} ."
            inputs_t = processor(images=rgb_crop, text=prompt_target, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_t = gdino_detector(**inputs_t)
            results_t = processor.post_process_grounded_object_detection(
                outputs_t, inputs_t.input_ids, threshold=0.15, text_threshold=0.2, target_sizes=[crop.shape[:2]]
            )[0]

            import re
            def normalize(s):
                return re.sub(r'[^a-zA-Z0-9]', '', s.lower())

            best_det_label = "Empty"
            best_det_conf = 0.0
            is_match = False
            
            # Find the competitive winner
            max_conf_c = 0.0
            best_label_c = "Empty"
            if len(results_c["scores"]) > 0:
                for i, box in enumerate(results_c["boxes"].cpu().numpy()):
                    if calculate_iou([box[0]+crop_x1, box[1]+crop_y1, box[2]+crop_x1, box[3]+crop_y1], [x1_gt, y1_gt, x2_gt, y2_gt]) > 0.2:
                        conf = float(results_c["scores"][i])
                        if conf > max_conf_c:
                            max_conf_c = conf
                            raw_lab = results_c["labels"][i].lower()
                            # Map to your specific label
                            for cls in unique_classes:
                                if normalize(cls) in normalize(raw_lab) or normalize(raw_lab) in normalize(cls):
                                    best_label_c = cls
                                    break

            # Find the targeted confidence
            target_conf = 0.0
            if len(results_t["scores"]) > 0:
                for i, box in enumerate(results_t["boxes"].cpu().numpy()):
                    if calculate_iou([box[0]+crop_x1, box[1]+crop_y1, box[2]+crop_x1, box[3]+crop_y1], [x1_gt, y1_gt, x2_gt, y2_gt]) > 0.2:
                        conf = float(results_t["scores"][i])
                        if conf > target_conf:
                            target_conf = conf

            # Final Logic:
            # 1. We trust the Competitive Winner (best_label_c) as the primary truth.
            #    If it finds a different brand from your list, it's a MISMATCH.
            if best_label_c != "Empty":
                best_det_label = best_label_c
                best_det_conf = max_conf_c
                is_match = (normalize(best_label_c) == normalize(gt['label']))
            
            # 2. FALLBACK: If competitive pass is uncertain, but targeted pass (Fanta) 
            #    is strong, we trust the targeted pass.
            elif target_conf > 0.3:
                is_match = True
                best_det_label = gt['label']
                best_det_conf = target_conf
            else:
                best_det_label = "Empty"
                best_det_conf = 0.0
                is_match = False

            if best_det_label != "Empty":
                all_detections_for_viz.append({"box": [x1_gt, y1_gt, x2_gt, y2_gt], "label": best_det_label, "score": best_det_conf})
                
            # Slot result remains at lines 202+

            slot_results.append({
                "label": gt['label'],
                "detected": best_det_label,
                "match": is_match,
                "confidence": best_det_conf,
                "expected_box": {"x1": x1_gt, "y1": y1_gt, "x2": x2_gt, "y2": y2_gt}
            })
            
            # Memory Management: Clear cache between crops to prevent OOM on limited hardware
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            
        
        # Annotate the frame
        annotated_frame = frame.copy()
        
        # Draw all found detections in green
        for det in all_detections_for_viz:
            b = [int(x) for x in det['box']]
            cv2.rectangle(annotated_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{det['label']} {det['score']:.2f}", (b[0], max(b[1]-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw Expected boxes in Blue/Red
        for res in slot_results:
            gt = res['expected_box']
            x1, y1, x2, y2 = int(gt['x1']), int(gt['y1']), int(gt['x2']), int(gt['y2'])
            color = (0, 255, 0) if res['match'] else (0, 0, 255) # Green if match, Red if mismatch
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            status = "OK" if res['match'] else f"ERR: {res['detected']}"
            cv2.putText(annotated_frame, f"{res['label']}: {status}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated_frame, slot_results, all_detections_for_viz

    if is_image:
        frame = cv2.imread(temp_path)
        if frame is None:
            return {"error": "Failed to read uploaded image"}
            
        annotated_frame, slot_results, detections = process_frame_targeted(frame, ground_truth)
        
        output_filename = f"out_{video.filename}"
        output_path = os.path.join("static", output_filename)
        cv2.imwrite(output_path, annotated_frame)
        
        final_results = slot_results
        for det in detections:
            detected_json.append({
                "label": det['label'],
                "confidence": float(det['score']),
                "x1": float(det['box'][0]), "y1": float(det['box'][1]),
                "x2": float(det['box'][2]), "y2": float(det['box'][3])
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
                
            annotated_frame, slot_results, detections = process_frame_targeted(frame, ground_truth)
            out.write(annotated_frame)
            
            if not first_frame_processed:
                first_frame_processed = True
                final_results = slot_results
                for det in detections:
                    detected_json.append({
                        "label": det['label'],
                        "confidence": float(det['score']),
                        "x1": float(det['box'][0]), "y1": float(det['box'][1]),
                        "x2": float(det['box'][2]), "y2": float(det['box'][3])
                    })
                    
        cap.release()
        out.release()
        
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
