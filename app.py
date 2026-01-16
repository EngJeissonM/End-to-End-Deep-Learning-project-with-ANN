#This is a Streamlit web application for predicting customer churn using a pre-trained deep learning model.

import streamlit as st # web app framework neccessary for the app
import numpy as np
import tensorflow as tf # deep learning library
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # data preprocessing required for the categorical values
import pandas as pd
import pickle # for loading the saved encoders and scaler

# Load the trained model
model = tf.keras.models.load_model('model.h5') 

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0]) # Selectbox for geography using categories from the one-hot encoder
gender = st.selectbox('Gender', label_encoder_gender.classes_) # Selectbox for Gender using classes from the label encoder
age = st.slider('Age', 18, 92) # Slider for age
balance = st.number_input('Balance') # Number input for balance
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4) # Slider for number of products
has_cr_card = st.selectbox('Has Credit Card', [0, 1]) # Selectbox for Has Credit Card
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({ # Create a DataFrame for input data
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    #here is missing Geography which will be one-hot encoded later
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# Create DataFrame for one-hot encoded geography
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography'])) 

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data, this step is crucial for the model to make accurate predictions
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0] # Probability of churn
st.write(f'Churn Probability: {prediction_proba:.2f}')# Display the churn probability

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

