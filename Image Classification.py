import os
import shutil
import cv2
import numpy as np

# Path to your segmented images
source_directory = r"C:\Users\sanga\Downloads\Segment_Mask"  # Update this path with your folder path
# Path to where you want to save classified images
melanoma_directory = r"C:\Users\sanga\Downloads\Image_Classification\melanoma"  # Update this path with your folder path
benign_directory = r"C:\Users\sanga\Downloads\Image_Classification\benign"  # Update this path with your folder path

# Define thresholds for classifying the image based on grayscale intensity
INTENSITY_THRESHOLD = 100  # Example threshold value, adjust based on your dataset

# Function to classify images based on intensity threshold
def classify_image_by_intensity(image_path):
    # Open image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    
    # Apply threshold to classify based on intensity
    _, thresholded_img = cv2.threshold(img, INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Calculate the mean intensity of the image after thresholding (can be a simple heuristic)
    mean_intensity = np.mean(thresholded_img)
    
    # Classify based on mean intensity
    if mean_intensity > 127:  # This is an arbitrary threshold, adjust based on your data
        return "melanoma"
    else:
        return "benign"


# Function to classify and move the image to the respective folder
def classify_and_move_image(image_path):
    # Classify the image based on intensity threshold
    class_name = classify_image_by_intensity(image_path)
    if class_name is None:
        return
    
    print(f"Classifying {image_path} as {class_name}")
    
    # Create target class directory if it doesn't exist
    if class_name == "melanoma":
        target_directory = melanoma_directory
    else:
        target_directory = benign_directory
    
    # Ensure target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    # Move the image to the respective directory
    filename = os.path.basename(image_path)
    target_path = os.path.join(target_directory, filename)
    
    # Move file using shutil.move
    shutil.move(image_path, target_path)
    print(f"Moved {filename} to {target_directory}")


# Function to process all images in the source directory
def classify_images_in_directory(source_directory):
    # Loop through all images in the source directory
    for root, dirs, files in os.walk(source_directory):
        for file_name in files:
            image_path = os.path.join(root, file_name)

            # Skip directories or non-image files
            if not os.path.isfile(image_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skipping non-image file: {file_name}")
                continue
            
            print(f"Processing image: {image_path}")
            
            # Classify and move the image
            classify_and_move_image(image_path)


# Run the function to classify and move images
classify_images_in_directory(source_directory)