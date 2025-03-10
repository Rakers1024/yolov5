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
def process_frame(frame, boxes):
    img_height, img_width = frame.shape[:2]
    
    # Process each box
    results_dict = {}
    for i, box in enumerate(boxes):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = normalize_to_pixel(box, img_height, img_width)
        
        # Crop image according to box
        cropped = frame[y1:y2, x1:x2]
        
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Box {i+1}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, results_dict

if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Read boxes
    boxes_path = "web/Boxes.json"
    boxes = read_boxes(boxes_path)
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        processed_frame, results = process_frame(frame, boxes)
        
        # Display results
        cv2.imshow('Real-time Detection', processed_frame)
        
        # Print detection results
        print("\nDetection Results:")
        for obj, count in results.items():
            print(f"{obj}: {count}")
            
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()