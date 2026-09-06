# DeepVision-Lab: End-to-End Multi-Model CIFAR-10 Classifier

An end-to-end image classification project using CIFAR-10 to compare traditional machine learning, artificial neural networks, and convolutional neural networks. The project includes model experimentation, performance comparison, model selection, and Streamlit deployment.

## Project Overview

The objective of this project is to build and evaluate multiple classification models for recognizing images across the 10 CIFAR-10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Multiple models were developed and compared to understand the effectiveness of different approaches for image classification. The final CNN model achieved **86.98% accuracy** and was deployed as an interactive Streamlit application.

## Models Implemented

| Model | Accuracy |
|---|---:|
| Logistic Regression | 39.80% |
| Artificial Neural Network (ANN) | 48.37% |
| Baseline CNN | 79.16% |
| Improved CNN | Higher than baseline |
| Augmented CNN (Final Model) | **86.98%** |

The Augmented CNN was selected as the final model based on its classification performance.

## Methodology

### 1. Data Preparation
- Loaded the CIFAR-10 dataset using TensorFlow/Keras.
- Normalized pixel values to the range 0–1.
- Prepared image data according to the requirements of each model.
- Used a stratified training-validation split for CNN experimentation.

### 2. Model Development
- Built a Logistic Regression baseline using flattened image features.
- Built an ANN using fully connected layers and dropout.
- Developed a baseline CNN for image classification.
- Improved the CNN using:
  - Convolutional layers
  - Batch Normalization
  - Dropout
  - Data augmentation
  - Learning-rate scheduling
  - Early stopping
  - Model checkpointing

### 3. Final Model

The selected model is an augmented CNN trained using image transformations such as:

- Horizontal flipping
- Width shifting
- Height shifting

The best-performing model was saved as:

`best_exp2_cnn.keras`

## Deployment

The final model was integrated into a Streamlit application that allows users to:

- Upload an image for classification
- Use sample CIFAR-10 images
- View the predicted class
- View the top 3 predicted classes
- Understand the image preprocessing and prediction process

### Live Application

**Streamlit:** https://cifar-10-image-classifier-hgf.streamlit.app/

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pillow
- Streamlit
- Jupyter Notebook
- Git & GitHub

## Project Structure

```text
DeepVision-Lab-End-to-End-Multi-Model-CIFAR-10-Classifier/
│
├── cifar10_examples/
│   ├── airplane.png
│   ├── automobile.png
│   ├── bird.png
│   ├── cat.png
│   ├── deer.png
│   ├── dog.png
│   ├── frog.png
│   ├── horse.png
│   ├── ship.png
│   └── truck.png
│
├── app.py
├── best_exp2_cnn.keras
├── readme.md
├── requirements.txt
└── .gitignore 

Key Outcome
The project demonstrates the progression from traditional machine learning to deep learning for image classification. The final augmented CNN achieved 86.98% accuracy, significantly outperforming the Logistic Regression, ANN, and baseline CNN approaches.

Note

The model is trained specifically on CIFAR-10 and therefore classifies images into one of the ten CIFAR-10 categories. Images outside these categories may still be assigned to the closest available class because the model is designed as a closed-set classifier.
