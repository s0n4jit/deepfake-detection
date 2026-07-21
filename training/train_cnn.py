import os
import matplotlib.pyplot as plt
import json
import time
import argparse
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
args, unknown = parser.parse_known_args()

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

class DeepfakeDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["path"]
        label = 0 if rec["label"] == "REAL" else 1
        
        # Load image via PIL to match torchvision transforms expectations
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train_cnn():
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
        
    train_recs = split["train"]
    test_recs = split["test"]
    
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
    
    train_dataset = DeepfakeDataset(train_recs, train_transform)
    test_dataset = DeepfakeDataset(test_recs, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load pretrained ResNet18 backbone
    print("Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    start_time = time.time()
    
    history_loss = []
    history_train_acc = []
    history_test_acc = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Evaluate test set accuracy at end of epoch
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_epoch_acc = test_correct / test_total
        
        history_loss.append(epoch_loss)
        history_train_acc.append(epoch_acc * 100.0)
        history_test_acc.append(test_epoch_acc * 100.0)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc * 100:.2f}% - Test Acc: {test_epoch_acc * 100:.2f}%")
        
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds")
    
    # Evaluate model on test set for final metrics
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
    
    # Evaluate train set accuracy too (for overfitting gap check)
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.numpy())
            
    train_acc = accuracy_score(train_labels, train_preds)
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
        
    # Generate run plots dynamically
    os.makedirs("docs", exist_ok=True)
    epochs_range = range(1, EPOCHS + 1)
    
    # 1. Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs_range, history_train_acc, label='Train Accuracy', color='#6366f1', marker='o')
    ax1.plot(epochs_range, history_test_acc, label='Test Accuracy', color='#10b981', marker='s')
    ax1.set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    
    ax2.plot(epochs_range, history_loss, label='Train Loss', color='#ef4444', marker='o')
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    curve_path = "docs/resnet18_training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved run training curves to {curve_path}")
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=len(test_dataset))
    ax.set_title("RESNET18 Confusion Matrix", fontsize=12, fontweight='bold', pad=12)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['REAL', 'FAKE'])
    ax.set_yticklabels(['REAL', 'FAKE'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > (len(test_dataset) / 4) else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')
            
    plt.tight_layout()
    cm_path = "docs/resnet18_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved run confusion matrix to {cm_path}")
    
    # 3. Model Comparison Bar Chart (Dynamically update Current Run metrics)
    runs = [
        "CNN Run 1\n(ResNet18)", "CNN Run 2\n(ResNet18)", "CNN Run 3\n(ResNet18)", 
        "CNN Run 4\n(ResNet18)", "CNN Run 5\n(Current Run)", "CNN Run 6\n(EffNet)"
    ]
    
    accuracy = [68.60, 71.90, 69.42, 71.10, test_acc * 100.0, 76.45]
    f1_score = [74.50, 74.60, 74.83, 71.26, f1 * 100.0, 78.81]
    
    x = np.arange(len(runs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width/2, accuracy, width, label='Test Accuracy', color='#6366f1')
    rects2 = ax.bar(x + width/2, f1_score, width, label='Test F1-Score', color='#10b981')
    
    ax.set_title('CNN Experimental Run Comparisons', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Metrics Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=0, fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    comparison_path = "docs/model_comparison_bar_chart.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Saved run model comparison bar chart to {comparison_path}")
    
    # Export state dict for deployment
    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_EXPORT_PATH)
    print(f"Saved trained CNN model state dict to {MODEL_EXPORT_PATH}")

if __name__ == "__main__":
    train_cnn()
