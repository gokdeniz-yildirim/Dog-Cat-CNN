import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from model import MultiObjectDogCatCNN
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize prediction on a single image')
    parser.add_argument('--model', type=str, default='multi_object_dog_cat_cnn.pth', help='Path to the trained model')
    parser.add_argument('--image', type=str, default='catdogp2.jpg', help='Path to the image file')
    return parser.parse_args()

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = MultiObjectDogCatCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def visualize_prediction(image_path, model, transform):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        class_output, bbox_output = model(input_tensor)

    predicted_class = torch.argmax(class_output, dim=1).item()
    predicted_bbox = bbox_output[0].tolist()

    draw = ImageDraw.Draw(image)
    width, height = image.size

    draw_bbox(draw, predicted_bbox, width, height, color="red", label="Predicted")

    font = ImageFont.load_default()
    class_name = 'Dog' if predicted_class == 1 else 'Cat'
    confidence = torch.nn.functional.softmax(class_output, dim=1)[0][predicted_class].item()
    draw.text((10, 10), f"Predicted: {class_name} ({confidence:.2f})", fill="red", font=font)

    return image

def draw_bbox(draw, bbox, width, height, color, label):
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin*width, ymin*height, xmax*width, ymax*height], outline=color, width=2)
    draw.text([xmin*width, ymin*height-15], label, fill=color)

def main():
    args = parse_args()
    try:
        model = load_model(args.model)
        transform = get_transform()

        image_with_bbox = visualize_prediction(args.image, model, transform)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_bbox)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()