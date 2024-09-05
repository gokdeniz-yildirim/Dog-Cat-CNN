import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import os
import xml.etree.ElementTree as ET
from model import MultiObjectDogCatCNN
from dataset import MultiObjectDogCatDataset
from utils import parse_args

def load_model(model_path):
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

def visualize_prediction(image, xml_path, model, transform):
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    # Assuming the output is a single tensor with 5 elements (1 for class, 4 for bbox)
    predicted_class = torch.argmax(output[0, :1]).item()
    predicted_bbox = output[0, 1:].tolist()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    true_class = 1 if root.find('object/name').text.lower() == 'dog' else 0
    bbox = root.find('object/bndbox')
    true_bbox = [
        float(bbox.find('xmin').text) / int(root.find('size/width').text),
        float(bbox.find('ymin').text) / int(root.find('size/height').text),
        float(bbox.find('xmax').text) / int(root.find('size/width').text),
        float(bbox.find('ymax').text) / int(root.find('size/height').text)
    ]

    draw = ImageDraw.Draw(image)
    width, height = image.size

    draw_bbox(draw, predicted_bbox, width, height, color="red", label="Predicted")
    draw_bbox(draw, true_bbox, width, height, color="green", label="True")

    font = ImageFont.load_default()
    draw.text((10, 10), f"Predicted: {'Dog' if predicted_class == 1 else 'Cat'}", fill="red", font=font)
    draw.text((10, 30), f"True: {'Dog' if true_class == 1 else 'Cat'}", fill="green", font=font)

    return image

def draw_bbox(draw, bbox, width, height, color, label):
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin*width, ymin*height, xmax*width, ymax*height], outline=color, width=2)
    draw.text([xmin*width, ymin*height-15], label, fill=color)

def main():
    args = parse_args()
    model = load_model("multi_object_dog_cat_cnn.pth")
    transform = get_transform()

    dataset = MultiObjectDogCatDataset(args.test_annotations, args.test_images)

    num_images = 5
    indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    for i, idx in enumerate(indices):
        item = dataset[idx]
        
        if not isinstance(item, tuple) or len(item) != 2:
            print(f"Warning: Unexpected item format at index {idx}")
            continue
        
        image, xml_path = item
        
        if not isinstance(image, Image.Image):
            print(f"Warning: Unexpected type for image at index {idx}")
            continue
        
        if not isinstance(xml_path, str):
            print(f"Warning: Unexpected type for xml_path at index {idx}. Type: {type(xml_path)}")
            continue

        image_with_bbox = visualize_prediction(image, xml_path, model, transform)
        axes[i].imshow(image_with_bbox)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
