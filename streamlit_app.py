import streamlit as st
import joblib
import pandas as pd

# Load trained model and label encoders
model = joblib.load("logistic_regression_playtennis_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Play Tennis Predictor", layout="centered")

st.title("🎾 Play Tennis Prediction App")
st.write("Predict whether to play tennis based on weather conditions.")

# User input fields (feature columns only)
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

    # Encode inputs using the same encoders used in training
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Decode prediction using target encoder
    result = label_encoders["Play Tennis"].inverse_transform([prediction])[0]

    # Display result
    if result.lower() == "yes":
        st.success("✅ Yes — Play Tennis")
    else:
        st.error("❌ No — Don't Play Tennis")
