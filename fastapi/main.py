from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import supervision as sv
import json
import os
import torch
import re
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

print("Loading Grounding DINO model...")
local_model_path = "./local_gdino_model"

if os.path.exists(local_model_path):
    print(f"Loading from local directory: {local_model_path}")
    model_id = local_model_path
else:
    print("Local directory not found. Loading from HuggingFace hub (will be cached)...")
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
        Processes the frame by running Grounding DINO once on the full high-resolution image,
        and then mapping the detected objects back to the planogram slots using IoU.
        """
        h, w = int(frame.shape[0]), int(frame.shape[1])
        print(f"DEBUG: Processing full frame of size {w}x{h}")
        
        slot_results = []
        all_detections_for_viz = []
        
        # 1. Clean class labels and generate a high-recall competitive prompt
        def get_clean_prompt(label):
            cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', label)
            return " ".join(cleaned.split())

        def normalize(s):
            return re.sub(r'[^a-zA-Z0-9]', '', s.lower())

        # Construct the full competitive prompt
        prompt_comp = " . ".join([get_clean_prompt(c) for c in unique_classes]).lower() + " ."
        
        # 2. Run Grounding DINO on the full frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb_frame, text=prompt_comp, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = gdino_detector(**inputs)
            
        import inspect
        sig = inspect.signature(processor.post_process_grounded_object_detection)
        kwargs = {"text_threshold": 0.2, "target_sizes": [[h, w]]}
        if "box_threshold" in sig.parameters:
            kwargs["box_threshold"] = 0.15
        else:
            kwargs["threshold"] = 0.15

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )[0]

        # 3. Hybrid HSV Brand Classification Engine
        def classify_hsv_color(h_val, s_val, v_val):
            if v_val < 50: 
                return "black"
            if s_val < 50: 
                if v_val > 200:
                    return "white"
                else:
                    return "grey"
            if h_val < 9 or h_val > 165:
                return "red"
            elif h_val < 24:
                return "orange"
            elif h_val < 38:
                return "yellow"
            elif h_val < 88:
                return "green"
            elif h_val < 133:
                return "blue"
            else:
                return "purple"

        def parse_colors_from_label(label_str):
            label_lower = label_str.lower()
            categories = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "grey"]
            matched = []
            for cat in categories:
                if cat in label_lower:
                    matched.append(cat)
            return matched

        class_colors = {cls: parse_colors_from_label(cls) for cls in unique_classes}

        def get_crop_classification(crop_img):
            default_cls = unique_classes[0] if unique_classes else "sprite (green)"
            if crop_img.size == 0:
                return default_cls
                
            cw_crop, ch_crop = crop_img.shape[1], crop_img.shape[0]
            ccx1 = int(cw_crop * 0.20)
            ccy1 = int(ch_crop * 0.20)
            ccx2 = int(cw_crop * 0.80)
            ccy2 = int(ch_crop * 0.80)
            
            center_crop = crop_img[ccy1:ccy2, ccx1:ccx2]
            hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            h_plane, s_plane, v_plane = hsv_crop[:,:,0], hsv_crop[:,:,1], hsv_crop[:,:,2]
            
            flat_h = h_plane.flatten()
            flat_s = s_plane.flatten()
            flat_v = v_plane.flatten()
            
            pixel_colors = [classify_hsv_color(flat_h[idx], flat_s[idx], flat_v[idx]) for idx in range(len(flat_h))]
            unique_p_colors, counts = np.unique(pixel_colors, return_counts=True)
            color_distribution = dict(zip(unique_p_colors, counts))
            
            non_bg_pixels = sum([counts[i] for i in range(len(unique_p_colors)) if unique_p_colors[i] not in ["white", "grey"]])
            if non_bg_pixels == 0:
                non_bg_pixels = len(flat_h)
                
            best_match_cls = default_cls
            max_score = -1.0
            for cls_key in unique_classes:
                expected_colors = class_colors[cls_key]
                if not expected_colors:
                    continue
                    
                match_pixels = sum([color_distribution.get(c, 0) for c in expected_colors])
                score = match_pixels / float(non_bg_pixels)
                
                if score > max_score:
                    max_score = score
                    best_match_cls = cls_key
                    
            return best_match_cls

        # 4. Map DINO detections back to planogram slots and classify
        for gt in gt_boxes:
            x1_gt = float(gt['x1'])
            y1_gt = float(gt['y1'])
            x2_gt = float(gt['x2'])
            y2_gt = float(gt['y2'])
            
            gt_w = x2_gt - x1_gt
            gt_h = y2_gt - y1_gt
            
            best_det = None
            max_iou = 0.0
            
            if len(results["scores"]) > 0:
                for i, box in enumerate(results["boxes"].cpu().numpy()):
                    det_w = box[2] - box[0]
                    det_h = box[3] - box[1]
                    
                    # Spatial filtering
                    if det_w > gt_w * 2.0 or det_w < gt_w * 0.5:
                        continue
                    if det_h > gt_h * 2.0 or det_h < gt_h * 0.5:
                        continue
                        
                    iou = calculate_iou(box, [x1_gt, y1_gt, x2_gt, y2_gt])
                    if iou > max_iou:
                        max_iou = iou
                        best_det = {
                            "box": box,
                            "conf": float(results["scores"][i])
                        }
            
            # Hybrid Classification
            if best_det is not None and max_iou > 0.25:
                # Crop the DINO detected box
                bx1, by1, bx2, by2 = [int(val) for val in best_det["box"]]
                crop = frame[by1:by2, bx1:bx2]
                detected_label = get_crop_classification(crop)
                confidence = best_det["conf"]
            else:
                # Fallback: Crop the expected slot coordinate directly
                pad_w = int(gt_w * 0.10)
                pad_h = int(gt_h * 0.10)
                cx1 = max(0, int(x1_gt - pad_w))
                cy1 = max(0, int(y1_gt - pad_h))
                cx2 = min(w, int(x2_gt + pad_w))
                cy2 = min(h, int(y2_gt + pad_h))
                crop = frame[cy1:cy2, cx1:cx2]
                detected_label = get_crop_classification(crop)
                confidence = 0.85  # Default confidence for static slot classification
                
            class_confidences = {cls: 0.0 for cls in unique_classes}
            class_confidences[detected_label] = confidence
                
            is_match = (normalize(detected_label) == normalize(gt['label']))
            
            slot_results.append({
                "label": gt['label'],
                "detected": detected_label,
                "match": is_match,
                "confidence": confidence,
                "all_scores": class_confidences,
                "expected_box": {"x1": x1_gt, "y1": y1_gt, "x2": x2_gt, "y2": y2_gt}
            })
            
            all_detections_for_viz.append({
                "box": [x1_gt, y1_gt, x2_gt, y2_gt],
                "label": detected_label,
                "score": confidence,
                "all_scores": class_confidences
            })
                
            # Print the result
            print(f"Slot expected '{gt['label']}' -> Winner: {detected_label}, IoU: {max_iou:.4f}")
            
        # Memory Management: Clear cache between frames to prevent memory leaks
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
                "all_scores": det['all_scores'],
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
                        "all_scores": det['all_scores'],
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
