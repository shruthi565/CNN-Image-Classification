# 🖼️ CIFAR-10 Image Classification using CNN
A deep learning project that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project demonstrates image preprocessing, CNN model development, model training, evaluation, and visualization of results.

## 📌 Project Overview
The objective of this project is to classify images into one of the ten CIFAR-10 categories using a Convolutional Neural Network (CNN). The model is trained on the CIFAR-10 dataset and evaluated using accuracy, confusion matrix, and classification metrics.

## 🧠 Dataset
The project uses the **CIFAR-10** dataset provided by TensorFlow.

### Dataset Information
- Training Images: **50,000**
- Testing Images: **10,000**
- Image Size: **32 × 32**
- Color Channels: **RGB (3 Channels)**
- Number of Classes: **10**

### Classes
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck
- 
## 🚀 Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📂 Project Workflow
1. Load the CIFAR-10 dataset
2. Visualize sample images
3. Normalize image pixel values
4. Build a CNN model
5. Compile the model
6. Train the model
7. Evaluate model performance
8. Predict image classes
9. Generate confusion matrix
10. Display classification report

## 🏗️ CNN Architecture
- Input Layer (32 × 32 × 3)
- Conv2D (32 Filters, 3×3, ReLU)
- MaxPooling2D
- Conv2D (64 Filters, 3×3, ReLU)
- MaxPooling2D
- Flatten Layer
- Dense Layer (64 neurons, ReLU)
- Output Layer (10 neurons, Softmax)

## 📊 Evaluation Metrics
The model is evaluated using:
- Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Confusion Matrix
- Classification Report
  - Precision
  - Recall
  - F1-Score

## ▶️ Installation
Clone the repository
```bash
git clone https://github.com/your-username/cifar10-image-classification-cnn.git
cd cifar10-image-classification-cnn
```
Install dependencies
```bash
pip install -r requirements.txt

## ▶️ Run the Project
```bash
python cifar10_cnn.py


