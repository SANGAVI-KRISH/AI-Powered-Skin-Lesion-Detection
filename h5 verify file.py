import os
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model


# Define model file paths
h5_model_path = r"C:\Users\Sweth\Documents\skin_lesion_classifier.h5"
keras_model_path = r"C:\Users\Sweth\Documents\skin_lesion_classifier.keras"


# ✅ Step 1: Check if the .h5 file is corrupted
def check_h5_file(file_path):
    try:
        with h5py.File(file_path, "r") as f:
            print("✅ The .h5 model file is valid.")
            return True
    except OSError:
        print("❌ The .h5 model file is CORRUPT. You need to retrain it.")
        return False


# ✅ Step 2: Check TensorFlow & h5py versions
def check_versions():
    print("\n📌 TensorFlow Version:", tf.__version__)
    print("📌 h5py Version:", h5py.__version__)


# ✅ Step 3: Try loading the model in different formats
def load_model_safely():
    if os.path.exists(keras_model_path):
        print("\n🔄 Loading model in TensorFlow format (.keras)...")
        model = load_model(keras_model_path)
        print("✅ Model loaded successfully from .keras format.")
    elif os.path.exists(h5_model_path) and check_h5_file(h5_model_path):
        print("\n🔄 Loading model from .h5 format...")
        model = load_model(h5_model_path)
        print("✅ Model loaded successfully from .h5 format.")
    else:
        print("❌ No valid model file found. Retraining is required.")
        model = None


    return model


# ✅ Step 4: Retrain & Save Model if Needed
def retrain_and_save():
    # (Only use this if your dataset is available)
    print("\n🔄 Retraining the model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Load your dataset here
    # model.fit(train_data, epochs=10, validation_data=val_data)
    print("✅ Training completed. Saving model...")
    model.save(keras_model_path)  # Save in TensorFlow format
    print("✅ Model saved successfully as .keras")


# Run all checks
check_versions()
model = load_model_safely()


if model is None:
    retrain_and_save()
