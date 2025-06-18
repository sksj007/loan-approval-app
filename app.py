import streamlit as st
import pandas as pd
from data_loading import load_data, preprocess_data
from model_training import train_model

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

st.title("Loan Approval Predictor")
st.write("Enter applicant details to predict default risk.")

amount = st.number_input("Loan Amount", min_value=0)
duration = st.number_input("Loan Duration (months)", min_value=1)
age = st.number_input("Applicant Age", min_value=18, max_value=100)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'amount': [amount],
        'duration': [duration],
        'age': [age],
        'status_A12': [0],
        'purpose_A43': [1],
    })
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'✅ Approved' if prediction == 1 else '❌ Rejected'}")
