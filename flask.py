from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


app = Flask(__name__)


# Load the trained model (Use the latest saved model)
MODEL_PATH = r"skin_lesion_classifier.keras"
model = load_model(MODEL_PATH)


# Define class labels
CLASS_LABELS = {0: "🟢 Benign", 1: "🔴 Melanoma"}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400


    file = request.files['file']
    img_path = "temp.jpg"
    file.save(img_path)


    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    # Predict
    prediction = model.predict(img_array)[0][0]
    result = CLASS_LABELS[1] if prediction > 0.5 else CLASS_LABELS[0]
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 4)


    return f"Prediction: {result} (Confidence: {confidence})"


if __name__ == '__main__':
    app.run(debug=True)
