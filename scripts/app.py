import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Load scaler and selected features
# The scaler is used to normalize the input features (important for many machine learning models)
scaler = pickle.load(open('../models/scaler.pk1', 'rb'))

# Load the list of selected features that the model expects
with open('../models/selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

# Define the numeric columns that will be input by the user
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Title for the Streamlit app
st.title("💓 Heart Disease Prediction App (Local Model)")

# Numeric inputs: Age, RestingBP, Cholesterol, Fasting Blood Sugar, MaxHR, and Oldpeak
age = st.number_input("Age", min_value=1, max_value=120)
resting_bp = st.number_input("RestingBP")
chol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
max_hr = st.number_input("MaxHR")
oldpeak = st.number_input("Oldpeak")

# Categorical inputs: Sex, Chest Pain Type, Resting ECG, Exercise Angina, and ST Slope
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
chest_pain = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'LVH', 'ST'])
exercise_angina = st.selectbox("Exercise Angina", ['N', 'Y'])
st_slope = st.selectbox("ST Slope", ['Up', 'Flat'])

# When the user clicks the "Predict" button, the prediction process is triggered
if st.button("Predict"):
    # Prepare numeric features from user input
    numeric_features = [age, resting_bp, chol, fasting_bs, max_hr, oldpeak]
    input_df = pd.DataFrame([numeric_features], columns=numeric_cols)

    # Scale numeric features using the pre-loaded scaler
    scaled_numeric = scaler.transform(input_df)

    # One-hot encode the categorical features manually (simulating pandas get_dummies with drop_first=True)
    chest_pain_encoded = [1 if chest_pain == 'ATA' else 0,
                          1 if chest_pain == 'NAP' else 0]

    resting_ecg_encoded = [1 if resting_ecg == 'LVH' else 0,
                           1 if resting_ecg == 'ST' else 0]

    sex_encoded = [sex]  # Already binary, so no encoding needed

    exercise_angina_encoded = [1 if exercise_angina == 'Y' else 0]

    st_slope_encoded = [1 if st_slope == 'Flat' else 0,
                        1 if st_slope == 'Up' else 0]

    # Combine all the features into a single input vector
    input_vector = np.hstack([
        scaled_numeric.flatten(),
        sex_encoded,
        chest_pain_encoded,
        resting_ecg_encoded,
        exercise_angina_encoded,
        st_slope_encoded
    ])

    # List of all feature names, which will be expected by the model
    all_feature_names = numeric_cols + ['Sex', 'ChestPainType_ATA', 'ChestPainType_NAP',
                                        'RestingECG_LVH', 'RestingECG_ST',
                                        'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

    # Convert the input vector into a DataFrame with the correct feature names
    input_df_full = pd.DataFrame([input_vector], columns=all_feature_names)

    # Filter to include only the selected features that the model was trained on
    input_selected = input_df_full[selected_features].values

    # Load the trained model from a local file
    model = joblib.load("../models/best_model.pkl")

    # Predict the risk of heart disease based on the input features
    prediction = model.predict(input_selected)[0]

    # Display the prediction result to the user
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease!")  # Display error if high risk
    else:
        st.success("✅ Low risk of Heart Disease!")  # Display success if low risk
