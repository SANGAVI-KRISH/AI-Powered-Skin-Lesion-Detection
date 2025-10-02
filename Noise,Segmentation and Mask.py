import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
# Paths
image_directory = r"C:\Users\sanga\Downloads\Noise_Removed"
output_directory = r"C:\Users\sanga\Downloads\Segment_Mask"
# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)
# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
# Function to apply noise reduction
def remove_noise(img):
    img = cv2.medianBlur(img, 5)  # Median filter (for salt-and-pepper noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur (for smoothening)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Non-Local Means Denoising
    return img
# Function to apply contrast enhancement
def enhance_contrast(img):
    enhancer = ImageEnhance.Contrast(Image.fromarray(img))
    return np.array(enhancer.enhance(2))  # Contrast increased by factor of 2
# Function to segment image using adaptive thresholding
def segment_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return mask
# Function to apply mask to the original image
def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)
# Function to process images
def process_images(directory):
    for root, _, files in os.walk(directory):  
        for file_name in files:
            image_path = os.path.join(root, file_name)
            if file_name.lower().endswith(image_extensions):
                try:
                    # Open image
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")  # Ensure RGB format
                        img_array = np.array(img)
                        # Apply processing steps
                        denoised = remove_noise(img_array)
                        contrast_enhanced = enhance_contrast(denoised)
                        mask = segment_image(contrast_enhanced)
                        segmented_img = apply_mask(contrast_enhanced, mask)
                        # Convert array to image
                        final_img = Image.fromarray(segmented_img)
                        # Save processed image
                        save_path = os.path.join(output_directory, file_name)
                        final_img.save(save_path)
                        print(f"Processed and saved: {file_name}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"Skipping non-image file: {file_name}")
# Start processing images
process_images(image_directory)