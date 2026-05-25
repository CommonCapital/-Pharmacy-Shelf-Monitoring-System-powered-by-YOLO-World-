from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import supervision as sv
import json
import os
import torch
import re
import base64
import asyncio
from pydantic import BaseModel
from typing import List, Optional
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

# Global Pydantic Models for One-Shot Registration & Verification
class BoxInput(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str

class RegisterRequest(BaseModel):
    image: str  # Base64 encoded image
    boxes: List[BoxInput]

class HSVColor(BaseModel):
    h: float
    s: float
    v: float
    dominant_color: str

class TemplateInput(BaseModel):
    label: str
    aspect_ratio: float
    hsv_color: HSVColor

class DetectRequest(BaseModel):
    image: str  # Base64 encoded image
    templates: List[TemplateInput]

# Global HSV Classification and Extraction Helpers
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

def extract_hsv_features(crop_img):
    """
    Extracts average H, S, V values and dominant color from a cropped BGR image center.
    """
    if crop_img is None or crop_img.size == 0:
        return {"h": 0.0, "s": 0.0, "v": 0.0, "dominant_color": "unknown"}
    
    h_orig, w_orig = crop_img.shape[:2]
    cx1 = int(w_orig * 0.2)
    cy1 = int(h_orig * 0.2)
    cx2 = int(w_orig * 0.8)
    cy2 = int(h_orig * 0.8)
    center_crop = crop_img[cy1:cy2, cx1:cx2]
    
    if center_crop.size == 0:
        center_crop = crop_img
        
    hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv_crop[:, :, 0]))
    s_mean = float(np.mean(hsv_crop[:, :, 1]))
    v_mean = float(np.mean(hsv_crop[:, :, 2]))
    
    flat_h = hsv_crop[:, :, 0].flatten()
    flat_s = hsv_crop[:, :, 1].flatten()
    flat_v = hsv_crop[:, :, 2].flatten()
    
    pixel_colors = [classify_hsv_color(flat_h[idx], flat_s[idx], flat_v[idx]) for idx in range(len(flat_h))]
    unique_p_colors, counts = np.unique(pixel_colors, return_counts=True)
    if len(counts) > 0:
        dominant_idx = np.argmax(counts)
        dominant_color = str(unique_p_colors[dominant_idx])
    else:
        dominant_color = "unknown"
        
    return {
        "h": round(h_mean, 2),
        "s": round(s_mean, 2),
        "v": round(v_mean, 2),
        "dominant_color": dominant_color
    }

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
async def process_video(
    video: UploadFile = File(...),
    config: str = Form(...),
    reference_image: Optional[UploadFile] = File(None)
):
    try:
        ground_truth = json.loads(config)
    except Exception as e:
        return {"error": f"Invalid config JSON: {str(e)}"}
        
    classes = [item['label'] for item in ground_truth]
    unique_classes = list(set(classes))
    
    text_prompt = ""
    if unique_classes:
        text_prompt = " . ".join(unique_classes).lower() + " ."

    # Extract templates dynamically from reference image if provided
    templates = []
    if reference_image:
        try:
            ref_bytes = await reference_image.read()
            ref_nparr = np.frombuffer(ref_bytes, np.uint8)
            ref_frame = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            if ref_frame is not None:
                ref_h, ref_w = ref_frame.shape[:2]
                for gt in ground_truth:
                    rx1 = max(0, int(float(gt['x1'])))
                    ry1 = max(0, int(float(gt['y1'])))
                    rx2 = min(ref_w, int(float(gt['x2'])))
                    ry2 = min(ref_h, int(float(gt['y2'])))
                    
                    ref_crop = ref_frame[ry1:ry2, rx1:rx2]
                    hsv_features = extract_hsv_features(ref_crop)
                    
                    det_w = rx2 - rx1
                    det_h = ry2 - ry1
                    aspect_ratio = float(det_w) / float(det_h) if det_h > 0 else 1.0
                    
                    templates.append({
                        "label": gt['label'],
                        "aspect_ratio": aspect_ratio,
                        "hsv_color": hsv_features
                    })
                print(f"DEBUG: Successfully extracted {len(templates)} templates from reference_image")
            else:
                print("DEBUG: Decoding reference_image returned None")
        except Exception as ex:
            print(f"ERROR: Failed to extract templates from reference_image: {ex}")

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

        # 3. Center crop classification function
        class_colors = {cls: parse_colors_from_label(cls) for cls in unique_classes}

        def get_crop_classification(crop_img):
            default_cls = unique_classes[0] if unique_classes else "unknown"
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

        def get_template_match_label(crop_img, expected_label):
            if not templates:
                return get_crop_classification(crop_img)
                
            default_cls = expected_label if expected_label in unique_classes else (unique_classes[0] if unique_classes else "unknown")
            if crop_img is None or crop_img.size == 0:
                return default_cls
                
            cw_crop, ch_crop = crop_img.shape[1], crop_img.shape[0]
            aspect_ratio = float(cw_crop) / float(ch_crop) if ch_crop > 0 else 1.0
            
            # Extract HSV features of current crop
            hsv_features = extract_hsv_features(crop_img)
            
            best_template = None
            best_score = -1.0
            
            for t in templates:
                # 1. Aspect Ratio similarity
                sim_geo = 1.0 - (abs(aspect_ratio - t["aspect_ratio"]) / max(aspect_ratio, t["aspect_ratio"]))
                # 2. Hue distance & similarity (Hue is circular [0, 180])
                dh = min(abs(hsv_features["h"] - t["hsv_color"]["h"]), 180.0 - abs(hsv_features["h"] - t["hsv_color"]["h"]))
                sim_h = 1.0 - (dh / 90.0) # Normalizes to [-1.0, 1.0], but we cap at 0
                sim_h = max(0.0, sim_h)
                # 3. Dominant Color match
                dom_match = 1.0 if hsv_features["dominant_color"] == t["hsv_color"]["dominant_color"] else 0.0
                
                # Weighted similarity score
                match_score = (sim_geo * 0.3) + (sim_h * 0.4) + (dom_match * 0.3)
                
                # Add a small tie-breaker boost to the expected label to avoid minor fluctuation flips
                if t["label"] == expected_label:
                    match_score += 0.05
                    
                if match_score > best_score:
                    best_score = match_score
                    best_template = t
                    
            if best_template is not None:
                return best_template["label"]
            return default_cls

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
            
            # Hybrid Classification and HSV extraction
            if best_det is not None and max_iou > 0.25:
                bx1, by1, bx2, by2 = [int(val) for val in best_det["box"]]
                crop = frame[by1:by2, bx1:bx2]
                detected_label = get_template_match_label(crop, gt['label'])
                confidence = best_det["conf"]
                hsv_features = extract_hsv_features(crop)
            else:
                pad_w = int(gt_w * 0.10)
                pad_h = int(gt_h * 0.10)
                cx1 = max(0, int(x1_gt - pad_w))
                cy1 = max(0, int(y1_gt - pad_h))
                cx2 = min(w, int(x2_gt + pad_w))
                cy2 = min(h, int(y2_gt + pad_h))
                crop = frame[cy1:cy2, cx1:cx2]
                detected_label = get_template_match_label(crop, gt['label'])
                confidence = 0.85
                hsv_features = extract_hsv_features(crop)
                
            class_confidences = {cls: 0.0 for cls in unique_classes}
            class_confidences[detected_label] = confidence
                
            is_match = (normalize(detected_label) == normalize(gt['label']))
            
            slot_results.append({
                "label": gt['label'],
                "detected": detected_label,
                "match": is_match,
                "confidence": confidence,
                "all_scores": class_confidences,
                "expected_box": {"x1": x1_gt, "y1": y1_gt, "x2": x2_gt, "y2": y2_gt},
                "hsv_color": hsv_features
            })
            
            all_detections_for_viz.append({
                "box": [x1_gt, y1_gt, x2_gt, y2_gt],
                "label": detected_label,
                "score": confidence,
                "all_scores": class_confidences,
                "hsv_color": hsv_features
            })
                
            print(f"Slot expected '{gt['label']}' -> Winner: {detected_label}, IoU: {max_iou:.4f}")
            
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        
        annotated_frame = frame.copy()
        
        for det in all_detections_for_viz:
            b = [int(x) for x in det['box']]
            cv2.rectangle(annotated_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{det['label']} {det['score']:.2f}", (b[0], max(b[1]-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for res in slot_results:
            gt = res['expected_box']
            x1, y1, x2, y2 = int(gt['x1']), int(gt['y1']), int(gt['x2']), int(gt['y2'])
            color = (0, 255, 0) if res['match'] else (0, 0, 255)
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
                "x2": float(det['box'][2]), "y2": float(det['box'][3]),
                "hsv_color": det.get('hsv_color')
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
                        "x2": float(det['box'][2]), "y2": float(det['box'][3]),
                        "hsv_color": det.get('hsv_color')
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


# NEW ENDPOINTS FOR ONE-SHOT REGISTRATION AND VERIFICATION

@app.post("/register-templates")
async def register_templates(req: RegisterRequest):
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode reference image"}
            
        h, w = frame.shape[:2]
        templates_out = []
        
        for box in req.boxes:
            x1 = max(0, int(box.x1))
            y1 = max(0, int(box.y1))
            x2 = min(w, int(box.x2))
            y2 = min(h, int(box.y2))
            
            crop = frame[y1:y2, x1:x2]
            hsv_features = extract_hsv_features(crop)
            
            det_w = x2 - x1
            det_h = y2 - y1
            aspect_ratio = float(det_w) / float(det_h) if det_h > 0 else 1.0
            
            templates_out.append({
                "label": box.label,
                "aspect_ratio": round(aspect_ratio, 3),
                "hsv_color": hsv_features
            })
            
        return {"templates": templates_out}
    except Exception as e:
        return {"error": str(e)}


@app.post("/detect")
async def detect_objects(req: DetectRequest):
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode target image"}
            
        h, w = frame.shape[:2]
        
        unique_labels = list(set([t.label for t in req.templates]))
        if not unique_labels:
            return {"detections": []}
            
        def get_clean_prompt(label):
            cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', label)
            return " ".join(cleaned.split())
            
        prompt_comp = " . ".join([get_clean_prompt(lbl) for lbl in unique_labels]).lower() + " ."
        
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
        
        detections_out = []
        
        if len(results["scores"]) > 0:
            for i, box in enumerate(results["boxes"].cpu().numpy()):
                bx1, by1, bx2, by2 = [max(0, int(val)) for val in box]
                bx2 = min(w, bx2)
                by2 = min(h, by2)
                
                crop = frame[by1:by2, bx1:bx2]
                hsv_features = extract_hsv_features(crop)
                
                det_w = bx2 - bx1
                det_h = by2 - by1
                aspect_ratio = float(det_w) / float(det_h) if det_h > 0 else 1.0
                
                best_template = None
                best_score = -1.0
                
                for t in req.templates:
                    sim_geo = 1.0 - (abs(aspect_ratio - t.aspect_ratio) / max(aspect_ratio, t.aspect_ratio))
                    dh = min(abs(hsv_features["h"] - t.hsv_color.h), 180.0 - abs(hsv_features["h"] - t.hsv_color.h))
                    sim_h = 1.0 - (dh / 90.0)
                    dom_match = 1.0 if hsv_features["dominant_color"] == t.hsv_color.dominant_color else 0.0
                    
                    match_score = (sim_geo * 0.3) + (sim_h * 0.4) + (dom_match * 0.3)
                    
                    if sim_geo >= 0.70 and dh <= 25.0:
                        if match_score > best_score:
                            best_score = match_score
                            best_template = t
                            
                if best_template is not None:
                    detections_out.append({
                        "name": best_template.label,
                        "confidence": float(results["scores"][i]),
                        "bbox": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3])
                        },
                        "hsv_color": hsv_features,
                        "match_score": round(best_score, 3)
                    })
                    
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
            
        return {"detections": detections_out}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/monitor/{shelf_id}")
async def websocket_monitor(websocket: WebSocket, shelf_id: str):
    await websocket.accept()
    print(f"WebSocket connected for shelf monitoring: {shelf_id}")
    
    try:
        init_data = await websocket.receive_json()
        
        # Read dynamic templates from websocket initialization
        raw_templates = init_data.get("templates", [])
        active_templates = []
        
        if raw_templates:
            for t in raw_templates:
                active_templates.append({
                    "label": t["label"],
                    "aspect_ratio": float(t["aspect_ratio"]) if t.get("aspect_ratio") is not None else 0.45,
                    "hsv_color": {
                        "h": float(t["hsv_color"]["h"]) if t["hsv_color"].get("h") is not None else 0.0,
                        "s": float(t["hsv_color"]["s"]) if t["hsv_color"].get("s") is not None else 0.0,
                        "v": float(t["hsv_color"]["v"]) if t["hsv_color"].get("v") is not None else 0.0,
                        "dominant_color": t["hsv_color"].get("dominant_color", "unknown")
                    }
                })
            print(f"WS: Dynamically initialized with {len(active_templates)} templates from client")
        else:
            # Fallback to mock templates using drug_names for backward compatibility
            drug_names = init_data.get("drug_names", [])
            print(f"WS fallback: Initialized with drug names: {drug_names}")
            for name in drug_names:
                color_lbl = "red"
                hsv_h = 5.0
                hsv_s = 220.0
                hsv_v = 180.0
                if "sprite" in name.lower() or "green" in name.lower():
                    color_lbl = "green"
                    hsv_h = 60.0
                    hsv_s = 200.0
                    hsv_v = 150.0
                elif "fanta" in name.lower() or "yellow" in name.lower() or "orange" in name.lower():
                    color_lbl = "yellow"
                    hsv_h = 25.0
                    hsv_s = 220.0
                    hsv_v = 200.0
                    
                active_templates.append({
                    "label": name,
                    "aspect_ratio": 0.45,
                    "hsv_color": {
                        "h": hsv_h,
                        "s": hsv_s,
                        "v": hsv_v,
                        "dominant_color": color_lbl
                    }
                })
        
        video_path = "yellow-filling.mp4"
        if not os.path.exists(video_path):
            video_path = "../yellow-filling.mp4"
            
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            print(f"WS: Streaming from video {video_path}")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                for _ in range(4):
                    cap.grab()
                    
                h, w = frame.shape[:2]
                unique_labels = list(set([t["label"] for t in active_templates]))
                
                if unique_labels:
                    prompt_comp = " . ".join(unique_labels).lower() + " ."
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = processor(images=rgb_frame, text=prompt_comp, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = gdino_detector(**inputs)
                        
                    results = processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, text_threshold=0.2, box_threshold=0.15, target_sizes=[[h, w]]
                    )[0]
                    
                    detections_out = []
                    if len(results["scores"]) > 0:
                        for i, box in enumerate(results["boxes"].cpu().numpy()):
                            bx1, by1, bx2, by2 = [max(0, int(val)) for val in box]
                            bx2 = min(w, bx2)
                            by2 = min(h, by2)
                            
                            crop = frame[by1:by2, bx1:bx2]
                            hsv_features = extract_hsv_features(crop)
                            
                            det_w = bx2 - bx1
                            det_h = by2 - by1
                            aspect_ratio = float(det_w) / float(det_h) if det_h > 0 else 1.0
                            
                            best_template = None
                            best_score = -1.0
                            
                            for t in active_templates:
                                sim_geo = 1.0 - (abs(aspect_ratio - t["aspect_ratio"]) / max(aspect_ratio, t["aspect_ratio"]))
                                dh = min(abs(hsv_features["h"] - t["hsv_color"]["h"]), 180.0 - abs(hsv_features["h"] - t["hsv_color"]["h"]))
                                sim_h = 1.0 - (dh / 90.0)
                                dom_match = 1.0 if hsv_features["dominant_color"] == t["hsv_color"]["dominant_color"] else 0.0
                                
                                match_score = (sim_geo * 0.3) + (sim_h * 0.4) + (dom_match * 0.3)
                                
                                if sim_geo >= 0.70 and dh <= 25.0:
                                    if match_score > best_score:
                                        best_score = match_score
                                        best_template = t
                                        
                            if best_template is not None:
                                detections_out.append({
                                    "name": best_template["label"],
                                    "confidence": float(results["scores"][i]),
                                    "bbox": {
                                        "x1": float(box[0]),
                                        "y1": float(box[1]),
                                        "x2": float(box[2]),
                                        "y2": float(box[3])
                                    },
                                    "hsv_color": hsv_features
                                })
                                
                    await websocket.send_json({"detections": detections_out})
                    
                await asyncio.sleep(0.3)
        else:
            print("WS: Video stream file not found, idling WS connection...")
            while True:
                await asyncio.sleep(1)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
