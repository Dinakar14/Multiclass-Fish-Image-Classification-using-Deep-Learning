# Multiclass-Fish-Image-Classification-using-Deep-Learning
To build a robust multiclass image classification system that can accurately identify fish species from images and deploy it as a user-friendly web application.

Project Overview

This project focuses on classifying fish images into multiple species using Deep Learning and Transfer Learning techniques.
The solution includes model training, evaluation, and deployment using Streamlit for real-time predictions.

The project compares:

A CNN model trained from scratch

A Transfer Learning model (EfficientNetB0)

and selects the best-performing model for deployment.

🎯 Problem Statement

To build a robust multiclass image classification system that can accurately identify fish species from images and deploy it as a user-friendly web application.

🧠 Skills & Technologies Used

Python

Deep Learning

TensorFlow / Keras

Transfer Learning (EfficientNetB0)

ImageDataGenerator & Data Augmentation

Model Evaluation Metrics

Streamlit (Deployment)

GitHub (Version Control)

🏗️ Project Architecture
Multiclass-Fish-Classification/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── cnn_model.h5
│   ├── best_transfer_model.h5
│   └── final_transfer_model.h5
│
├── train_cnn.py
├── train_transfer.py
├── evaluate_models.py
├── app.py
├── requirements.txt
└── README.md

📊 Dataset Description

Image dataset organized into folder-wise fish species

Loaded using TensorFlow ImageDataGenerator

Includes training and testing images

Data augmentation applied to improve generalization

🔄 Approach & Workflow
1️⃣ Data Preprocessing

Image resizing to 224×224

Normalization (rescale to 0–1)

Data augmentation:

Rotation

Zoom

Horizontal flip

Width & height shifts

2️⃣ Model Training
🔹 CNN from Scratch

Custom convolutional layers

MaxPooling

Fully connected layers

Dropout for regularization

🔹 Transfer Learning (EfficientNetB0)

Pretrained on ImageNet

Fine-tuning of deeper layers

Class-imbalance handled using class weights

Optimized using Adam optimizer

3️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Evaluation results are generated using:

evaluate_models.py

4️⃣ Model Deployment (Streamlit)

The final model is deployed using Streamlit with features:

Upload fish image

Predict fish category

Display confidence score

🚀 How to Run the Project
🔹 Step 1: Clone Repository
git clone https://github.com/your-username/Multiclass-Fish-Classification.git
cd Multiclass-Fish-Classification

🔹 Step 2: Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Train Models
python train_cnn.py
python train_transfer.py

🔹 Step 5: Evaluate Model
python evaluate_models.py

🔹 Step 6: Run Streamlit App
python -m streamlit run app.py

📈 Results & Insights

Transfer Learning outperformed CNN from scratch

Class imbalance significantly affected predictions

Solved using:

Class weighting

Fine-tuning pretrained layers

Final model shows improved generalization across fish species

🧪 Challenges Faced

Class imbalance in dataset

Label ambiguity in certain classes

Early convergence to dominant class

✅ Solutions Implemented

Class-weighted loss

Aggressive fine-tuning

Enhanced classifier head

Learning rate optimization
