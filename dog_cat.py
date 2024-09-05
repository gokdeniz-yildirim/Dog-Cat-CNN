import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MultiObjectDogCatDataset, get_transform
from model import MultiObjectDogCatCNN
from train import train_model
from predict import predict_and_visualize
from utils import parse_args, set_seed, get_device, save_model, load_model

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args)

    print(f"Mode: {args.mode}")
    print(f"Using device: {device}")

    if args.mode == 'train':
        print("Training mode:")
        print(f"Training annotations directory: {args.annotations}")
        print(f"Training images directory: {args.images}")
        print(f"Validation annotations directory: {args.test_annotations}")
        print(f"Validation images directory: {args.test_images}")

        train_dataset = MultiObjectDogCatDataset(args.annotations, args.images, transform=get_transform(is_train=True), num_boxes=args.num_boxes, max_objects=args.num_boxes)
        val_dataset = MultiObjectDogCatDataset(args.test_annotations, args.test_images, transform=get_transform(is_train=False), num_boxes=args.num_boxes, max_objects=args.num_boxes)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = MultiObjectDogCatCNN(num_classes=args.num_classes, num_boxes=args.num_boxes, grid_size=7).to(device)
        print(f"Model created and moved to device: {device}")
        
        # Calculate and print the output shape of the feature extractor
        input_shape = (3, 224, 224)  # Assuming input images are 224x224 RGB
        output_shape = model.calculate_output_shape(input_shape)
        print(f"Feature extractor output shape: {output_shape}")
        
        trained_model = train_model(model, train_loader, val_loader, args)
        save_model(trained_model, args.model_path)
        print(f"Model saved to: {args.model_path}")

    elif args.mode == 'predict':
        print("Prediction mode:")
        print(f"Test annotations directory: {args.test_annotations}")
        print(f"Test images directory: {args.test_images}")
        print(f"Model path: {args.model_path}")
        print(f"Output directory: {args.output_dir}")

        test_dataset = MultiObjectDogCatDataset(args.test_annotations, args.test_images, transform=get_transform(is_train=False), num_boxes=args.num_boxes, max_objects=args.num_boxes)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"Test dataset size: {len(test_dataset)}")

        model = MultiObjectDogCatCNN(num_classes=args.num_classes, num_boxes=args.num_boxes, grid_size=7).to(device)
        model = load_model(model, args.model_path)
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded and moved to device: {device}")

        predict_and_visualize(model, test_loader, device, args.confidence_threshold, args.output_dir)

    else:
        print("Invalid mode. Please choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()