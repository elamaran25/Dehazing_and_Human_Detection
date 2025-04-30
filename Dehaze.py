import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import matplotlib.pyplot as plt
from huggingface_hub import login

# Authenticate with Hugging Face
HF_TOKEN = "your_hf_token"  # Replace with your actual token
login(HF_TOKEN)

# Load YOLO model
print("Loading YOLO model...")
yolo = YOLO("yolov8n.pt")

# Load HuggingFace model for better classification
print("Loading classification model...")
hf_model_id = "facebook/deit-base-distilled-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(hf_model_id, token=HF_TOKEN)
classifier = AutoModelForImageClassification.from_pretrained(hf_model_id, token=HF_TOKEN)
target_labels = classifier.config.id2label

# Define keywords for categorization
human_keywords = ["person", "man", "woman", "boy", "girl", "human", "child", "adult", "people"]
animal_keywords = ["cat", "dog", "horse", "cow", "sheep", "bear", "bird", "animal", 
                  "mammal", "elephant", "lion", "tiger", "zebra", "giraffe"]

def categorize(label):
    """Categorize based on keywords in the label"""
    label = label.lower()
    if any(k in label for k in human_keywords):
        return "Human"
    elif any(k in label for k in animal_keywords):
        return "Animal"
    else:
        return "Other"

def classify_object(image):
    """Classify an image using the HuggingFace model"""
    # Ensure proper image size and format
    image = cv2.resize(image, (224, 224))
    image = Image.fromarray(image)
    
    # Extract features and get predictions
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = classifier(**inputs)
    
    # Get probabilities and find the most likely class
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get top 3 predictions for better decision making
    top_probs, top_indices = torch.topk(probs, k=3, dim=1)
    
    # Return the best label and its confidence
    label_id = top_indices[0][0].item()
    confidence = probs[0][label_id].item()
    label = target_labels[label_id]
    
    return label, confidence, top_probs[0][0].item()

def dehaze_image(image):
    """Apply dehazing to improve image quality"""
    # Simple contrast enhancement as a basic dehazing technique
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img

def process_frame(frame):
    """Process a single frame"""
    # Convert to RGB for model processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply dehazing to improve image quality
    clean_frame = dehaze_image(rgb_frame)
    
    # Set a higher confidence threshold for YOLO
    results = yolo.predict(clean_frame, conf=0.4, verbose=False)
    
    # Create a copy for drawing
    result_frame = frame.copy()
    
    # Track the detected objects for visualization
    detected_objects = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates and confidence from YOLO
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_conf = box.conf[0].item()
            yolo_class = r.names[int(box.cls[0].item())]
            
            # Skip if box is too small
            if (x2-x1) < 20 or (y2-y1) < 20:
                continue
                
            # Crop the object
            obj_crop = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
            
            if obj_crop.size == 0:
                continue
    
            try:
                # For real-time performance, we could skip secondary classification
                # and just use YOLO's output for faster processing
                if yolo_class in ['person'] or yolo_conf > 0.7:
                    # Just use YOLO's classification for better performance
                    category = categorize(yolo_class)
                    class_label = yolo_class
                    confidence = yolo_conf
                else:
                    # Only use the more expensive classifier for uncertain cases
                    class_label, confidence, top_confidence = classify_object(obj_crop)
                    category = categorize(class_label)
                    
                # Set color based on category
                if category == "Human":
                    color = (0, 255, 0)  # Green for humans
                elif category == "Animal":
                    color = (0, 165, 255)  # Orange for animals (BGR format)
                else:
                    color = (0, 0, 255)  # Red for others
                    
                # Draw rectangle and label
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{category}: {class_label} ({confidence:.2f})"
                
                # Add background to text for better visibility
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result_frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
                cv2.putText(result_frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error processing object: {e}")
    
    return result_frame

# Main function to start real-time detection with camera
def main():
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera initialized. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the result
        cv2.imshow('Real-time Detection', processed_frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()