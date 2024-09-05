import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class MultiObjectDogCatDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, transform=None, grid_size=7, num_boxes=2, num_classes=2, max_objects=5):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.file_list = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        xml_file = self.file_list[idx]
        xml_path = os.path.join(self.annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_file = root.find('filename').text
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        w, h = image.size
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            label = 1 if name == 'dog' else 0
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / w
            ymin = float(bbox.find('ymin').text) / h
            xmax = float(bbox.find('xmax').text) / w
            ymax = float(bbox.find('ymax').text) / h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes, 5 + self.num_classes))

        for i, (box, label) in enumerate(zip(boxes, labels)):
            if i >= self.max_objects:
                break
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]

            grid_x = int(self.grid_size * x_center)
            grid_y = int(self.grid_size * y_center)

            target[grid_y, grid_x, 0, 0:4] = torch.tensor([x_center, y_center, width, height])
            target[grid_y, grid_x, 0, 4] = 1  
            target[grid_y, grid_x, 0, 5 + label] = 1  

        return image, target

def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to slightly larger size
            transforms.RandomCrop((224, 224)),  # Then crop to target size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])