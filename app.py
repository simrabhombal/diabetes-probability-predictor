import json
import os
import pickle
import subprocess

import joblib
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

# Load Groq API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Load your trained model
# with open("random_forest.pkl", "rb") as f:
#     model = pickle.load(f)

model = joblib.load("random_forest.joblib")


# Streamlit App UI
st.title("🩺 Diabetes Prediction App")

st.markdown("Enter patient details below to predict diabetes risk:")

# Input features (adjust as per your dataset)
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=30, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=400, value=100)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
BMI = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# === Predict Button ===
if st.button("Predict Diabetes"):
    input_features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                BMI, DiabetesPedigreeFunction, Age]])
    
    # Predict using the model
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0][1]

    result_text = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.subheader(f"🧪 Prediction: **{result_text}**")
    st.write(f"Probability of being Diabetic is `{int(prediction_proba * 100)}%`")

    # === Enhance explanation using Ollama ===
    st.markdown("---")
    # st.subheader("🤖 AI Explanation (Ollama)")
    
    prompt = f"""A patient's diabetes test result is predicted as "{result_text}" with a probability of {(prediction_proba * 100):.2f}%.
    If the probability is more than 50%, tell the user to take care about their health and necesssary actions to be taken.
    Explain this result in simple and reassuring terms, considering the following values:
    - Pregnancies: {Pregnancies}
    - Glucose: {Glucose}
    - Blood Pressure: {BloodPressure}
    - Skin Thickness: {SkinThickness}
    - Insulin: {Insulin}
    - BMI: {BMI}
    - Diabetes Pedigree Function: {DiabetesPedigreeFunction}
    - Age: {Age}
    """

    # Call Groq with LLaMA 3
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a friendly medical assistant who explains test results in clear and calm terms."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 400
    }

    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=body)
        res_json = res.json()
        explanation = res_json['choices'][0]['message']['content']
        st.markdown("### 🤖 AI Explanation")
        st.write(explanation)
    except Exception as e:
        st.error(f"Failed to get explanation: {e}")  