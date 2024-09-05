import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import MultiObjectLoss
from torch.cuda.amp import GradScaler, autocast

def train_model(model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = model.to(device)
    print(f"Model moved to device: {device}")

    # Enable gradient checkpointing
    model.features.apply(lambda m: m.register_forward_hook(lambda m, _, output: output.requires_grad_(True)))

    criterion = MultiObjectLoss(bbox_weight=args.bbox_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = GradScaler()

    # Gradient accumulation steps
    accumulation_steps = 4  # Adjust this value as needed

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_bbox_loss = 0
        class_correct = 0
        class_total = 0

        optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                loss, bbox_loss, obj_loss, class_loss = criterion(output, target)
                loss = loss / accumulation_steps  # Normalize the loss

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            total_bbox_loss += bbox_loss.item()
            
            # Calculate accuracy
            predicted = output[..., 5:].argmax(dim=-1)
            target_class = target[..., 5:].argmax(dim=-1)
            class_total += target_class.numel()
            class_correct += (predicted == target_class).sum().item()

        # Perform the final optimization step for any remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        avg_bbox_loss = total_bbox_loss / len(train_loader)
        class_accuracy = 100. * class_correct / class_total

        model.eval()
        test_loss = 0
        test_bbox_loss = 0
        class_correct = 0
        class_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                with autocast():
                    output = model(data)
                    loss, bbox_loss, _, _ = criterion(output, target)
                test_loss += loss.item()
                test_bbox_loss += bbox_loss.item()
                
                # Calculate accuracy
                predicted = output[..., 5:].argmax(dim=-1)
                target_class = target[..., 5:].argmax(dim=-1)
                class_total += target_class.numel()
                class_correct += (predicted == target_class).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_bbox_loss = test_bbox_loss / len(test_loader)
        test_accuracy = 100. * class_correct / class_total

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Bounding Box Loss: {avg_bbox_loss:.4f}")
        print(f"Train Accuracy: {class_accuracy:.2f}%")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Bounding Box Loss: {avg_test_bbox_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print()

        scheduler.step(avg_test_loss)

    return model