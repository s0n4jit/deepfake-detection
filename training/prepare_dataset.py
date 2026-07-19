import os
import json
import cv2
import dlib
import random
import shutil

# Paths
DATASET_DIR = "dataset"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
SPLIT_PATH = os.path.join(DATASET_DIR, "split.json")
SHAPE_PREDICTOR_PATH = "app/models/shape_predictor_68_face_landmarks.dat"

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def is_frontal_face(shape):
    """
    Check if a face is frontal using the symmetry of cheek and nose tip landmarks.
    Left cheek edge: landmark 0
    Right cheek edge: landmark 16
    Nose tip: landmark 30
    """
    x_left = shape.part(0).x
    x_right = shape.part(16).x
    x_nose = shape.part(30).x
    
    d_left = x_nose - x_left
    d_right = x_right - x_nose
    
    if d_left <= 0 or d_right <= 0:
        return False
        
    ratio = min(d_left, d_right) / max(d_left, d_right)
    return ratio >= 0.55

def process_dataset(folders, max_images=None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    real_images = []
    fake_images = []
    
    dropped_no_face = 0
    dropped_non_frontal = 0
    total_processed = 0
    
    for folder_name in folders:
        folder_path = os.path.join(DATASET_DIR, folder_name, folder_name)
        metadata_file = os.path.join(DATASET_DIR, f"metadata{int(folder_name[-2:] if folder_name[-2:].isdigit() else 0)}.json")
        
        if not os.path.exists(folder_path) or not os.path.exists(metadata_file):
            print(f"Skipping folder {folder_name} (not found)")
            continue
            
        print(f"Processing folder: {folder_name}...")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        for filename in os.listdir(folder_path):
            if not filename.endswith(".jpg"):
                continue
                
            total_processed += 1
            mp4_key = filename.replace(".jpg", ".mp4")
            if mp4_key not in metadata:
                continue
                
            label = metadata[mp4_key]["label"]
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            
            if len(rects) == 0:
                dropped_no_face += 1
                continue
                
            # If multiple faces, pick the largest one
            rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
            
            # Check landmarks for frontal pose
            shape = predictor(gray, rect)
            if not is_frontal_face(shape):
                dropped_non_frontal += 1
                continue
                
            # Crop face region with 10% padding
            h, w = img.shape[:2]
            l = max(0, rect.left() - int(0.1 * (rect.right() - rect.left())))
            t = max(0, rect.top() - int(0.1 * (rect.bottom() - rect.top())))
            r = min(w, rect.right() + int(0.1 * (rect.right() - rect.left())))
            b = min(h, rect.bottom() + int(0.1 * (rect.bottom() - rect.top())))
            
            face_crop = img[t:b, l:r]
            if face_crop.size == 0:
                continue
                
            # Resize face crop to standard 256x256
            face_crop = cv2.resize(face_crop, (256, 256))
            
            # Save cropped face to processed directory
            dest_filename = f"{folder_name}_{filename}"
            dest_path = os.path.join(PROCESSED_DIR, dest_filename)
            cv2.imwrite(dest_path, face_crop)
            
            record = {"path": dest_path, "label": label}
            if label == "REAL":
                real_images.append(record)
            else:
                fake_images.append(record)
                
            if max_images and (len(real_images) + len(fake_images)) >= max_images:
                break
                
    print(f"\nProcessing Complete:")
    print(f"Total files checked: {total_processed}")
    print(f"Dropped (no face): {dropped_no_face}")
    print(f"Dropped (non-frontal): {dropped_non_frontal}")
    print(f"Valid REAL images: {len(real_images)}")
    print(f"Valid FAKE images: {len(fake_images)}")
    
    # Class Balancing (Undersampling)
    min_count = min(len(real_images), len(fake_images))
    print(f"Balancing classes to {min_count} samples each...")
    
    random.seed(42)
    balanced_real = random.sample(real_images, min_count)
    balanced_fake = random.sample(fake_images, min_count)
    
    balanced_dataset = balanced_real + balanced_fake
    random.shuffle(balanced_dataset)
    
    # Train/Test Split (70/30)
    split_idx = int(0.7 * len(balanced_dataset))
    train_set = balanced_dataset[:split_idx]
    test_set = balanced_dataset[split_idx:]
    
    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")
    
    # Save split configuration
    split_data = {
        "train": train_set,
        "test": test_set
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Saved train/test split config to {SPLIT_PATH}")

if __name__ == "__main__":
    folders_to_use = ["DeepFake00", "DeepFake01", "DeepFake02", "DeepFake03", "DeepFake04"]
    process_dataset(folders_to_use)
