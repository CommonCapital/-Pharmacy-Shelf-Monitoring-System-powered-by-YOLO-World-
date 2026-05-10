import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import os

def verify_detection():
    print("Initializing YOLO-World model...")
    # Using 'l' version as specified in our architecture
    model = YOLOWorld(model_id="yolo_world/l")
    
    # Use the sample image already in the root
    image_path = "../dog.jpeg"
    if not os.path.exists(image_path):
        # fallback if run from root
        image_path = "dog.jpeg"
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Define some classes to test zero-shot capability
    # Even though it's a dog, we test if it can see 'person', 'backpack', 'dog'
    classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
    model.set_classes(classes)
    
    print(f"Running inference on {image_path} with classes: {classes}")
    results = model.infer(image)
    detections = sv.Detections.from_inference(results)
    
    print("\nDetection Results:")
    for i in range(len(detections.xyxy)):
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        label = classes[class_id]
        print(f"- Found {label} with confidence {confidence:.2f}")

    if len(detections.xyxy) > 0:
        print("\nStep 1 Verified: YOLO-World is detecting objects correctly.")
    else:
        print("\nStep 1 Failed: No objects detected.")

if __name__ == "__main__":
    verify_detection()
