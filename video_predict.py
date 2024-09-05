import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import MultiObjectDogCatCNN
from utils import load_model


def process_video(model_path, video_path, output_path, confidence_threshold=0.5, iou_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiObjectDogCatCNN(num_classes=2, num_boxes=5, grid_size=7)
    model = load_model(model, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(input_tensor)

        boxes, class_ids, confidences = process_predictions(predictions[0], confidence_threshold, iou_threshold)

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            label = "Dog" if class_id == 1 else "Cat"
            color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]))

        out.write(frame)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_predictions(predictions, conf_threshold, iou_threshold):
    grid_size, _, num_boxes, _ = predictions.shape
    boxes = []
    class_ids = []
    confidences = []

    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(num_boxes):
                box = predictions[i, j, b, :4]
                conf = predictions[i, j, b, 4]
                class_prob = predictions[i, j, b, 5:]
                class_id = torch.argmax(class_prob)
                max_prob = torch.max(class_prob)

                if conf * max_prob > conf_threshold:
                    x, y, w, h = box
                    x = (x + j) / grid_size
                    y = (y + i) / grid_size
                    w = w / grid_size
                    h = h / grid_size
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(class_id.item())
                    confidences.append((conf * max_prob).item())

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    
    return [boxes[i] for i in indices], [class_ids[i] for i in indices], [confidences[i] for i in indices]

if __name__ == "__main__":
    model_path = "multi_object_detector.pth"
    video_path = "input_video.mp4"
    output_path = "output_video.mp4"
    process_video(model_path, video_path, output_path)