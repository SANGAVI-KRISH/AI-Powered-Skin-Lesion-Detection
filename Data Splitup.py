import os
import shutil
import random

# Define paths
base_dir = r"C:\Users\sanga\Downloads\Image_Classification"  # Your original dataset
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Define class names
categories = ["melanoma", "benign"]

# Split function with error handling and image format filtering
def split_data(category, split_ratio=0.8):
    source_folder = os.path.join(base_dir, category)
    train_folder = os.path.join(train_dir, category)
    val_folder = os.path.join(val_dir, category)

    # Check if the source folder exists and contains images
    if not os.path.exists(source_folder):
        print(f"Source folder for category {category} does not exist!")
        return
    
    # Filter only image files with specific extensions
    images = [img for img in os.listdir(source_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check if there are any valid images in the folder
    if not images:
        print(f"No images found in the folder {source_folder} for category {category}!")
        return
    
    # Ensure target directories exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Shuffle images and split them into training and validation sets
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))
    for img in val_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

    print(f"{category}: {len(train_images)} train, {len(val_images)} validation")

# Apply splitting for each category
for category in categories:
    split_data(category)

print("✅ Dataset split completed successfully!")