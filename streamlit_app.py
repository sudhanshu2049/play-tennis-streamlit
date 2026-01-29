import streamlit as st
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("logistic_regression_playtennis_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Play Tennis Predictor", layout="centered")

st.title("🎾 Play Tennis Prediction")
st.write("Predict whether to play tennis based on weather conditions.")

# Input widgets (ONLY feature columns)
outlook = st.selectbox("Outlook", label_encoders["Outlook"].classes_)
temperature = st.selectbox("Temperature", label_encoders["Temperature"].classes_)
humidity = st.selectbox("Humidity", label_encoders["Humidity"].classes_)
wind = st.selectbox("Wind", label_encoders["Wind"].classes_)

if st.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame({
        "Outlook": [outlook],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind": [wind]
    })

    # Encode input using SAME encoders as training
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]

    # Decode output using Play Tennis encoder
    result = label_encoders["Play Tennis"].inverse_transform([prediction])[0]

    if result.lower() == "y
