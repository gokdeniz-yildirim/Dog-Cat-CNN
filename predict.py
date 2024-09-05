import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from model import MultiObjectDogCatCNN
from utils import parse_args, get_device, load_model, non_max_suppression

def predict(model, image_path, device, confidence_threshold=0.5, nms_threshold=0.4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    
    output = output.squeeze(0).cpu()
    boxes = output[..., :4]
    scores = output[..., 4]
    class_scores = output[..., 5:]

    # Convert to [x1, y1, x2, y2] format
    boxes[..., 0] -= boxes[..., 2] / 2
    boxes[..., 1] -= boxes[..., 3] / 2
    boxes[..., 2] += boxes[..., 0]
    boxes[..., 3] += boxes[..., 1]

    
    boxes[..., [0, 2]] *= original_width
    boxes[..., [1, 3]] *= original_height

   
    keep = non_max_suppression(boxes.view(-1, 4), scores.view(-1), nms_threshold)

    detections = []
    for i in keep:
        box = boxes.view(-1, 4)[i]
        score = scores.view(-1)[i]
        class_score, class_pred = class_scores.view(-1, 2)[i].max(0)
        if score > confidence_threshold:
            label = "Dog" if class_pred == 1 else "Cat"
            detections.append((label, score.item(), *box.tolist()))

    return detections

def draw_boxes(image_path, detections):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    for label, confidence, x1, y1, x2, y2 in detections:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label}: {confidence:.2f}", fill="red")
    
    return image

def predict_and_visualize(model, test_loader, device, confidence_threshold, nms_threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image, _) in enumerate(test_loader):
        image_path = test_loader.dataset.file_list[i]
        detections = predict(model, os.path.join(test_loader.dataset.images_dir, image_path), device, confidence_threshold, nms_threshold)
        result_image = draw_boxes(os.path.join(test_loader.dataset.images_dir, image_path), detections)
        
        output_path = os.path.join(output_dir, f"prediction_{i}.png")
        result_image.save(output_path)
        print(f"Saved prediction for image {i} to {output_path}")

def main():
    args = parse_args()
    device = get_device(args)

    model = MultiObjectDogCatCNN(num_classes=args.num_classes, num_boxes=args.num_boxes).to(device)
    model = load_model(model, args.model_path)

    image_path = input("Enter the path to the image: ")
    detections = predict(model, image_path, device, args.confidence_threshold, args.nms_threshold)
    
    result_image = draw_boxes(image_path, detections)
    result_image.show()

if __name__ == "__main__":
    main()
