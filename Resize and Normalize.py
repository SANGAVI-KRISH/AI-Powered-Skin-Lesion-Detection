import os
from PIL import Image
import numpy as np


# Path where original images are stored
image_directory = r'C:\Users\sanga\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2'
# Path where resized images will be saved (new folder for all resized images)
output_directory = r'C:\Users\sanga\Downloads\Resized_Images'


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
                        # Resize the image
                        resized_img = img.resize((224, 224))
                       
                        # Convert image to numpy array
                        img_array = np.array(resized_img)


                        # Normalize image to [0, 1]
                        img_array = img_array / 255.0


                        # Alternatively, you can normalize to [-1, 1]
                        # img_array = (img_array / 127.5) - 1


                        # Convert the normalized array back to an image
                        normalized_img = Image.fromarray((img_array * 255).astype(np.uint8))


                        # Save normalized image in the output folder
                        save_path = os.path.join(output_directory, file_name)
                        normalized_img.save(save_path)
                        print(f"Processed and saved {file_name} to {save_path}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"Skipping non-image file: {file_name}")


# Start processing images
process_images(image_directory)
