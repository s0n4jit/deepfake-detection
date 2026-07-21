import os
import json
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Configuration
SPLIT_PATH = "dataset/split.json"
MODEL_EXPORT_PATH = "app/models/cnn_model_v1.pt"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train CNN model on GPU/CPU.")
parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"],
                    help="Device to use for training (cuda, cpu, auto)")
parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"],
                    help="CNN backbone to use (resnet18, efficientnet_b0)")
parser.add_argument("--unfreeze_blocks", type=int, default=0, choices=[0, 1, 2],
                    help="Number of last blocks to unfreeze (0=none, 1=last block, 2=last 2 blocks)")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (0.0 to 0.9)")
parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization weight decay")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of balanced samples to train on (0 = all)")
parser.add_argument("--output", type=str, default="app/models/cnn_model_v1.pt", help="Path to save trained model")
args, unknown = parser.parse_known_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
MODEL_EXPORT_PATH = args.output

# Set device
if args.device == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Warning: CUDA is not available on this system. Please make sure GPU drivers and CUDA-enabled PyTorch are installed.")
        print("To install PyTorch with CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("To install PyTorch with CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("Falling back to CPU...")
        device = torch.device("cpu")
elif args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

NEW_DATASET_DIR = "dataset/archive/Dataset"

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=0):
        self.transform = transform
        self.samples = []
        
        # We explicitly map Real -> 0, Fake -> 1
        real_dir = os.path.join(root_dir, "Real")
        fake_dir = os.path.join(root_dir, "Fake")
        
        real_list = []
        fake_list = []
        
        if os.path.exists(real_dir):
            print(f"Scanning Real images in {real_dir}...")
            for f in os.listdir(real_dir):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    real_list.append((os.path.join(real_dir, f), 0))
                    
        if os.path.exists(fake_dir):
            print(f"Scanning Fake images in {fake_dir}...")
            for f in os.listdir(fake_dir):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    fake_list.append((os.path.join(fake_dir, f), 1))
                    
        # Apply max_samples balancing if specified
        if max_samples > 0:
            half = max_samples // 2
            random.seed(42) # For reproducible subsampling
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[:half]
            fake_list = fake_list[:half]
            
        self.samples = real_list + fake_list
        random.shuffle(self.samples)
        print(f"Total loaded: {len(self.samples)} images from {root_dir}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def train_cnn():
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Data Augmentation for training, simple resize/normalize for testing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    print("Loading datasets...")
    train_dataset = DeepfakeDataset(os.path.join(NEW_DATASET_DIR, "Train"), train_transform, max_samples=args.max_samples)
    test_dataset = DeepfakeDataset(os.path.join(NEW_DATASET_DIR, "Test"), test_transform, max_samples=args.max_samples // 3 if args.max_samples > 0 else 0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load pretrained backbone
    print(f"Loading pretrained {args.backbone}...")
    if args.backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze blocks
        if args.unfreeze_blocks >= 1:
            print("Unfreezing layer4 of ResNet18...")
            for param in model.layer4.parameters():
                param.requires_grad = True
        if args.unfreeze_blocks >= 2:
            print("Unfreezing layer3 of ResNet18...")
            for param in model.layer3.parameters():
                param.requires_grad = True
                
        # Replace the classification head
        num_features = model.fc.in_features
        if args.dropout > 0:
            print(f"Applying dropout of {args.dropout} to classification head...")
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(num_features, 2)
            )
        else:
            model.fc = nn.Linear(num_features, 2)
            
    elif args.backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze blocks
        # EfficientNet-B0 has 9 blocks inside model.features (0 to 8)
        if args.unfreeze_blocks >= 1:
            print("Unfreezing features[8] of EfficientNet-B0...")
            for param in model.features[8].parameters():
                param.requires_grad = True
        if args.unfreeze_blocks >= 2:
            print("Unfreezing features[7] of EfficientNet-B0...")
            for param in model.features[7].parameters():
                param.requires_grad = True
                
        # Replace classification head
        num_features = model.classifier[1].in_features
        if args.dropout > 0:
            print(f"Applying dropout of {args.dropout} to classification head...")
            model.classifier = nn.Sequential(
                nn.Dropout(p=args.dropout, inplace=True),
                nn.Linear(num_features, 2)
            )
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.0, inplace=True),
                nn.Linear(num_features, 2)
            )
            
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay
    )
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                running_acc = correct / total
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Running Loss: {loss.item():.4f} - Running Acc: {running_acc*100:.2f}%")
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc * 100:.2f}%")
        
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    # Evaluate model on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    inf_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    inf_duration = time.time() - inf_start
    avg_inf_time = inf_duration / len(test_dataset)
    
    # Use final epoch's running training accuracy to avoid extremely slow full pass over 140k images
    train_acc = epoch_acc
    test_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nResults:")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    print(f"Average Inference Time per image: {avg_inf_time * 1000.0:.2f} ms")
    
    # Log VRAM usage
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU VRAM allocated: {vram_allocated:.2f} MB")
        
    # Export state dict for deployment
    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)
    # Save the model state dict plus weight parameters
    torch.save(model.state_dict(), MODEL_EXPORT_PATH)
    print(f"Saved trained CNN model state dict to {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    train_cnn()
