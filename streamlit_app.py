import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("logistic_regression_playtennis_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Play Tennis Predictor", layout="centered")

st.title("🎾 Play Tennis Prediction App")
st.write("Enter weather conditions to predict whether to play tennis.")

# Input fields
outlook = st.selectbox("Outlook", label_encoders["Outlook"].classes_)
temperature = st.selectbox("Temperature", label_encoders["Temperature"].classes_)
humidity = st.selectbox("Humidity", label_encoders["Humidity"].classes_)
wind = st.selectbox("Wind", label_encoders["Wind"].classes_)

if st.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame({
        "Outlook": [outlook],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind": [wind]
    })

    # Encode input
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Yes, Play Tennis!")
    else:
        st.error("❌ No, Don't Play Tennis.")

