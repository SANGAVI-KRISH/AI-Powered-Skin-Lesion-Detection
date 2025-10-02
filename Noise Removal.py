import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from skimage import filters


# Path where original images are stored
image_directory = r"C:\Users\sanga\Downloads\Resized_Images"
# Path where resized images will be saved (new folder for all resized images)
output_directory = r'C:\Users\sanga\Downloads\Noise_Removed'


# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)


# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


# Function to process images
def process_images(directory):
    for root, dirs, files in os.walk(directory):  # Recursively go through all directories
        for file_name in files:
            image_path = os.path.join(root, file_name)


            # Check if the file is a valid image (based on extension)
            if file_name.lower().endswith(image_extensions):
                try:
                    with Image.open(image_path) as img:
                        # Convert to numpy array
                        img_array = np.array(img)


                        # Noise removal using median filter (works well for salt-and-pepper noise)
                        denoised_img = cv2.medianBlur(img_array, 5)


                        # Additional Noise Removal (Gaussian Blur)
                        denoised_img_gaussian = cv2.GaussianBlur(denoised_img, (5, 5), 0)


                        # Additional Noise Removal (Non-Local Means Denoising)
                        denoised_img_nlmeans = cv2.fastNlMeansDenoisingColored(denoised_img_gaussian, None, 10, 10, 7, 21)


                        # Contrast enhancement using PIL's ImageEnhance (you can adjust the factor)
                        enhancer = ImageEnhance.Contrast(Image.fromarray(denoised_img_nlmeans))
                        contrast_img = enhancer.enhance(2)  # Increase contrast by a factor of 2 (adjustable)


                        # Convert back to numpy array for further processing
                        contrast_array = np.array(contrast_img)


                        # Convert the image to grayscale (for segmentation)
                        gray_img = cv2.cvtColor(contrast_array, cv2.COLOR_RGB2GRAY)


                        # Thresholding for mask creation (simple segmentation approach)
                        _, mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)


                        # Apply mask to segment the lesion
                        segmented_img = cv2.bitwise_and(contrast_array, contrast_array, mask=mask)


                        # Normalize image to [0, 1]
                        normalized_img = segmented_img / 255.0


                        # Convert the normalized array back to an image
                        final_img = Image.fromarray((normalized_img * 255).astype(np.uint8))


                        # Save the processed image in the output folder
                        save_path = os.path.join(output_directory, file_name)
                        final_img.save(save_path)
                        print(f"Processed and saved {file_name} to {save_path}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"Skipping non-image file: {file_name}")


# Start processing images
process_images(image_directory)