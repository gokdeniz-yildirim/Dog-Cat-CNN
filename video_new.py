import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import MultiObjectDogCatCNN
from utils import non_max_suppression
import os
import numpy as np

def process_video(video_path, model, device, confidence_threshold=0.01, output_path=None, frame_interval=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        
        input_tensor = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(input_tensor)
        
        predictions = predictions.squeeze(0).cpu().numpy()
        
        # Reshape predictions
        grid_size, _, num_boxes, box_attrs = predictions.shape
        predictions = predictions.reshape(grid_size * grid_size * num_boxes, box_attrs)
        
        # Filter out low confidence predictions
        mask = predictions[:, 4] > confidence_threshold
        filtered_predictions = predictions[mask]
        
        print(f"Frame {frame_count}: Number of predictions above threshold: {len(filtered_predictions)}")
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw a test rectangle and text
        cv2.rectangle(vis_frame, (50, 50), (100, 100), (0, 255, 0), 2)
        cv2.putText(vis_frame, "Test", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if len(filtered_predictions) > 0:
            boxes = filtered_predictions[:, :4]
            scores = filtered_predictions[:, 4]
            class_ids = filtered_predictions[:, 5:].argmax(axis=1)
            
            # Convert YOLO format to (x1, y1, x2, y2)
            boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * frame.shape[1]
            boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * frame.shape[0]
            boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) * frame.shape[1]
            boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) * frame.shape[0]
            
            # Apply NMS
            keep_boxes = non_max_suppression(boxes, scores, iou_threshold=0.5)
            
            for box, score, class_id in zip(boxes[keep_boxes], scores[keep_boxes], class_ids[keep_boxes]):
                x1, y1, x2, y2 = box.astype(int)
                color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'Dog' if class_id == 1 else 'Cat'}: {score:.2f}"
                cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                print(f"Frame {frame_count}: Box: {(x1,y1,x2,y2)}, Score: {score:.2f}, Class: {'Dog' if class_id == 1 else 'Cat'}")
        
        # Display the frame with predictions
        cv2.imshow('Video Processing', vis_frame)
        
        # Write the frame to output video if specified
        if output_path:
            out.write(vis_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Increased wait time for slower playback
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = MultiObjectDogCatCNN(num_classes=2, num_boxes=5, grid_size=7).to(device)
        model.load_state_dict(torch.load('multi_object_dog_cat_cnn.pth', map_location=device))
        model.eval()
        print("Model loaded successfully.")

        input_video = 'dogcatvid1.mp4'
        output_video = 'dogcatvid1outputss.mp4'
        
        print(f"Processing video: {input_video}")
        process_video(input_video, model, device, confidence_threshold=0.01, output_path=output_video, frame_interval=2)
        print(f"Video processing complete. Output saved to: {output_video}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()