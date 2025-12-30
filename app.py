import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encoder_furnishingstatus.pkl', 'rb') as file:
    onehot_encoder_furnishingstatus = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

with open('scaler_y.pkl', 'rb') as file:
    scaler_y = pickle.load(file)


st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter house details below")

with st.form("house_form"):
    area = st.number_input("Area (sqft)", min_value=100, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
    stories = st.number_input("Stories", min_value=1, max_value=5, step=1)

    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, step=1)
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

    submit = st.form_submit_button("Predict Price")

if submit:
    data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": int(label_encoder['mainroad'].transform([mainroad])[0]),
        "guestroom": int(label_encoder['guestroom'].transform([guestroom])[0]),
        "basement": int(label_encoder['basement'].transform([basement])[0]),
        "hotwaterheating": int(label_encoder['hotwaterheating'].transform([hotwaterheating])[0]),
        "airconditioning": int(label_encoder['airconditioning'].transform([airconditioning])[0]),
        "parking": parking,
        "prefarea": int(label_encoder['prefarea'].transform([prefarea])[0]),
    }
    data_df = pd.DataFrame([data])
    furnishing_encoded = onehot_encoder_furnishingstatus.transform([[furnishingstatus]])

    furnishingstatus_df = pd.DataFrame(
    furnishing_encoded,
    columns=onehot_encoder_furnishingstatus.get_feature_names_out(['furnishingstatus'])
    )

    final_df = pd.concat([data_df, furnishingstatus_df], axis=1)

    inpot_data = scalar.transform(final_df)

    prediction_scaled = model.predict(inpot_data)

    # Inverse transform to get actual price
    predicted_price = scaler_y.inverse_transform(prediction_scaled)[0][0]
    st.success(f"💰 Predicted House Price: ₹{predicted_price:,.2f}")