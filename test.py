from ultralytics import YOLO
import json
import cv2
import numpy as np

# Load the model
model = YOLO("yolo11s.pt")

# Read bounding boxes from JSON file
def read_boxes(json_path):
    with open(json_path, 'r') as f:
        boxes = json.load(f)
    return boxes

# Convert normalized coordinates to pixel coordinates
def normalize_to_pixel(box, img_height, img_width):
    x1 = float(box['x1']) * img_width
    y1 = float(box['y1']) * img_height
    x2 = float(box['x2']) * img_width
    y2 = float(box['y2']) * img_height
    return int(x1), int(y1), int(x2), int(y2)

# Main process
def process_image(image_path, boxes_path):
    # Read image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    
    # Read boxes
    boxes = read_boxes(boxes_path)
    
    # Process each box
    results_dict = {}
    for i, box in enumerate(boxes):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = normalize_to_pixel(box, img_height, img_width)
        
        # Crop image according to box
        cropped = image[y1:y2, x1:x2]
        
        # Predict with the model
        results = model(cropped)
        
        # Draw results on the original image
        for result in results:
            # Get detection results
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            confs = result.boxes.conf
            
            # Count objects
            for name in set(names):
                if name not in results_dict:
                    results_dict[name] = 0
                results_dict[name] += names.count(name)
            
            # Draw box on original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Box {i+1}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save results
    cv2.imwrite('web/output.png', image)
    
    # Print detection results
    print("\nDetection Results:")
    for obj, count in results_dict.items():
        print(f"{obj}: {count}")

    return results_dict

if __name__ == "__main__":
    image_path = "web/test.png"
    boxes_path = "web/Boxes.json"
    process_image(image_path, boxes_path)
    
