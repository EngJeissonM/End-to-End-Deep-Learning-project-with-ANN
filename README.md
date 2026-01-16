 # End-to-End Deep Learning Project with ANN
Bank Customer Churn Prediction
 # Project Overview

This project implements an end-to-end deep learning solution to predict customer churn in a banking context using an Artificial Neural Network (ANN).

The model is built with TensorFlow and Keras, includes feature engineering and preprocessing, and is deployed as a Streamlit web application for real-time predictions.

 # Problem Statement

Given customer and bank account information, the objective is to predict whether a customer will exit the bank.
This is a binary classification problem.

 # Dataset

File: Churn_Modelling.csv

Features: 11 customer and account-related variables

Target: Exited (1 = churn, 0 = retained)

 # Technologies Used

Python

TensorFlow & Keras

Pandas, NumPy, Scikit-learn

Streamlit

Jupyter Notebook

 # Project Structure
├── app.py
├── Churn_Modelling.csv
├── experiments.ipynb
├── feature_engineering.ipynb
├── prediction.ipynb
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
└── README.md

 # Running the Application
pip install -r requirements.txt
streamlit run app.py

 # Results

The ANN model provides accurate churn predictions and is accessible through a user-friendly Streamlit interface.

 # Author

Jeisson Morales
Industrial Engineer | Data Science & AI Enthusiast
