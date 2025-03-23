# AI-Powered-Skin-Lesion-Detection

# **DermaScan: AI-Powered Skin Lesion Classification**  
An AI-based system for classifying skin lesions as **benign** or **melanoma** using deep learning. Built with **TensorFlow, OpenCV, Flask, and Kaggle datasets**, this project streamlines the **detection of skin cancer** with image preprocessing, noise removal, segmentation, and CNN-based classification.  

## 🚀 **Project Overview**  
Skin cancer is one of the most common types of cancer, and **early detection** is crucial for treatment. This project implements a **Convolutional Neural Network (CNN)** to classify **skin lesions** using the **HAM10000 dataset**.  

## 🏗 **Features**  
✔ **Image Preprocessing** (Resizing, Normalization, Denoising)  
✔ **Lesion Segmentation & Masking**  
✔ **Deep Learning Model for Classification**  
✔ **Web Interface (Flask) for Image Upload & Prediction**  
✔ **Deployment-Ready Model for Real-World Use**  

---

## 📂 **Project Structure**  
```
├── dataset/                  # Skin Lesion Dataset (HAM10000)
│   ├── train/                # Training images (80%)
│   ├── validation/           # Validation images (20%)
│   ├── test/                 # Test images
├── preprocessing/            # Image Processing Scripts
│   ├── resize.py             # Resize and Normalize Images
│   ├── denoise.py            # Noise Removal & Contrast Enhancement
│   ├── segmentation.py       # Segmentation & Masking
├── model/                    # CNN Model Training & Testing
│   ├── train.py              # Model Training Script
│   ├── evaluate.py           # Model Evaluation
│   ├── skin_lesion_classifier.h5  # Trained Model
├── flask_app/                # Web Application (Flask)
│   ├── app.py                # Flask Backend
│   ├── templates/            # HTML Templates
│   ├── static/               # CSS, JS, Images
├── README.md                 # Project Documentation
```

---

## 📊 **Dataset**  
- **Dataset Name:** [HAM10000 (Kaggle)](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)  
- **Classes:**  
  - **Benign (Non-Cancerous)**
  - **Malignant (Melanoma - Cancerous)**  
- **Format:** `.jpg` images  
- **Size:** 10,000+ images  

---

## 🛠 **Tech Stack & Libraries**  
✅ **Programming Language:** Python  
✅ **Deep Learning:** TensorFlow, Keras  
✅ **Image Processing:** OpenCV, PIL, NumPy  
✅ **Web Framework:** Flask  
✅ **Dataset Handling:** Pandas, Kaggle API  

---

## 🏗 **Installation & Setup**  

### **🔹 Step 1: Clone the Repository**  
```bash
git clone https://github.com/yourusername/dermascan.git
cd dermascan
```

### **🔹 Step 2: Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **🔹 Step 3: Download Dataset**  
Use **Kaggle API** to download the dataset:  
```python
import kagglehub
kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

### **🔹 Step 4: Run Preprocessing Scripts**  
```bash
python preprocessing/resize.py
python preprocessing/denoise.py
python preprocessing/segmentation.py
```

### **🔹 Step 5: Train the Model**  
```bash
python model/train.py
```

### **🔹 Step 6: Run Flask App**  
```bash
python flask_app/app.py
```
Then open `http://127.0.0.1:5000/` in your browser.  

---

## 🎯 **Usage**  
1️⃣ **Upload a skin lesion image** via the web app  
2️⃣ **AI model processes the image** (denoising, segmentation, classification)  
3️⃣ **Get instant classification result** (Benign / Melanoma)  

---

## 📌 **Results & Model Performance**  
📌 **Accuracy:** ~92% on test data  
📌 **Precision / Recall Metrics**  
📌 **Confusion Matrix for Classification**  
