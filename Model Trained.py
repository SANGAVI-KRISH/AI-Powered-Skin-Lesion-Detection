import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"C:\Users\sanga\skin_lesion_classifier.keras")

# Define a confidence threshold
confidence_threshold = 0.8  # Lowered threshold for more lenient predictions (e.g., 60%)

# Function to apply preprocessing steps
def preprocess_image(img_path):
    # 1. Load the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Image not found at {img_path}.")
        return None
    
    # 2. Resize image to the expected input size (224x224 for many models like VGG, ResNet, etc.)
    img_resized = cv2.resize(img, (224, 224))
    
    # 3. Noise Removal using Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # 4. Convert to grayscale (if necessary for segmentation or masking)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    
    # 5. Thresholding to create a mask (this is just a simple example, replace with more advanced methods if needed)
    _, mask = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    
    # 6. Apply the mask to isolate regions (Optional, for example: lesion area isolation)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # 7. Normalize image for model input (scale pixel values between 0 and 1)
    img_array = img_masked / 255.0
    
    # 8. Add batch dimension (for model input)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict image
def predict_image(img_path):
    # Preprocess the image with resizing, noise removal, etc.
    img_array = preprocess_image(img_path)

    if img_array is None:
        return  # Exit if image couldn't be processed

    # Predict the class
    raw_prediction = model.predict(img_array)[0][0]

    # Print the raw prediction value (before applying any threshold)
    print(f"Raw prediction value: {raw_prediction:.4f}")

    # Manually check the model output (probabilities)
    melanoma_prob = raw_prediction  # Melanoma probability
    benign_prob = 1 - raw_prediction  # Benign probability (since it's binary classification)

    # Print out the probabilities
    print(f"Melanoma probability: {melanoma_prob:.4f}")
    print(f"Benign probability: {benign_prob:.4f}")

    # Check which class has the higher probability and whether it meets the confidence threshold
    if melanoma_prob > benign_prob:
        if melanoma_prob >= confidence_threshold:
            print(f"🔴 Predicted: Melanoma (Confidence: {melanoma_prob:.4f})")
        else:
            print("⚠️ Uncertain: Melanoma probability too low to confidently predict.")
    elif benign_prob > melanoma_prob:
        if benign_prob >= 0.5:
            print(f"🟢 Predicted: Benign (Confidence: {benign_prob:.4f})")
        else:
            print("⚠️ Uncertain: Benign probability too low to confidently predict.")
    else:
        print("⚠️ Uncertain: Both probabilities are equal or too low for a reliable prediction.")
    
    # Display the image with prediction (Optional)
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {'Melanoma' if melanoma_prob > benign_prob else 'Benign'}\nConfidence: {melanoma_prob if melanoma_prob > benign_prob else benign_prob:.4f}")
    plt.show()

# Main code (image path is predefined)
img_path = r"C:\Users\sanga\OneDrive\Documents\Pictures\SANGAVI_K_23I351.jpg"  # Replace with your desired image path
predict_image(img_path)
