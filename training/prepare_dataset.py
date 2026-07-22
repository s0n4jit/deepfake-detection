import os
import json
import random

# Paths
NEW_DATASET_DIR = "dataset/archive/Dataset"
SPLIT_PATH = "dataset/split.json"

def prepare_new_dataset():
    train_real_dir = os.path.join(NEW_DATASET_DIR, "Train", "Real")
    train_fake_dir = os.path.join(NEW_DATASET_DIR, "Train", "Fake")
    test_real_dir = os.path.join(NEW_DATASET_DIR, "Test", "Real")
    test_fake_dir = os.path.join(NEW_DATASET_DIR, "Test", "Fake")
    
    # Check if directories exist
    for d in [train_real_dir, train_fake_dir, test_real_dir, test_fake_dir]:
        if not os.path.exists(d):
            print(f"Error: Directory not found: {d}")
            return
            
    print("Listing files in directories...")
    train_real_files = [os.path.join(train_real_dir, f) for f in os.listdir(train_real_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_fake_files = [os.path.join(train_fake_dir, f) for f in os.listdir(train_fake_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_real_files = [os.path.join(test_real_dir, f) for f in os.listdir(test_real_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_fake_files = [os.path.join(test_fake_dir, f) for f in os.listdir(test_fake_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found in raw dataset:")
    print(f"  Train Real: {len(train_real_files)}")
    print(f"  Train Fake: {len(train_fake_files)}")
    print(f"  Test Real: {len(test_real_files)}")
    print(f"  Test Fake: {len(test_fake_files)}")
    
    # Settings for subsampling to be fast
    num_train_per_class = 1500
    num_test_per_class = 500
    
    random.seed(42)
    
    print(f"Subsampling {num_train_per_class} per class for training...")
    selected_train_real = random.sample(train_real_files, num_train_per_class)
    selected_train_fake = random.sample(train_fake_files, num_train_per_class)
    
    print(f"Subsampling {num_test_per_class} per class for testing...")
    selected_test_real = random.sample(test_real_files, num_test_per_class)
    selected_test_fake = random.sample(test_fake_files, num_test_per_class)
    
    # Create records
    train_records = []
    for p in selected_train_real:
        train_records.append({"path": p, "label": "REAL"})
    for p in selected_train_fake:
        train_records.append({"path": p, "label": "FAKE"})
        
    test_records = []
    for p in selected_test_real:
        test_records.append({"path": p, "label": "REAL"})
    for p in selected_test_fake:
        test_records.append({"path": p, "label": "FAKE"})
        
    # Shuffle splits
    random.shuffle(train_records)
    random.shuffle(test_records)
    
    print(f"Balanced Dataset Prepared:")
    print(f"  Train Set: {len(train_records)} images")
    print(f"  Test Set: {len(test_records)} images")
    
    split_data = {
        "train": train_records,
        "test": test_records
    }
    
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Saved split config successfully to {SPLIT_PATH}")

if __name__ == "__main__":
    prepare_new_dataset()
