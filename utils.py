import argparse
import torch
import numpy as np
from torchvision import transforms
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Object Dog and Cat Classification')
    parser.add_argument('--annotations', type=str, default='annotations', help='Path to training annotations directory')
    parser.add_argument('--images', type=str, default='images', help='Path to training images directory')
    parser.add_argument('--test_annotations', type=str, default='antt', help='Path to test annotations directory')
    parser.add_argument('--test_images', type=str, default='imgg', help='Path to test images directory')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--bbox_weight', type=float, default=1.0, help='weight for bounding box loss (default: 1.0)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes (default: 2)')
    parser.add_argument('--num_boxes', type=int, default=5, help='maximum number of objects per image (default: 5)')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='train or predict mode')
    parser.add_argument('--model_path', type=str, default='multi_object_dog_cat_cnn.pth', help='path to save/load the model')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='confidence threshold for predictions')
    parser.add_argument('--output_dir', type=str, default='predictions', help='directory to save prediction results')
    return parser.parse_args()

# ... rest of the file remains the same


def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device(args):
    return torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def non_max_suppression(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update indices
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Box: (x1, y1, x2, y2)
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter_area

    iou = inter_area / union
    return iou
