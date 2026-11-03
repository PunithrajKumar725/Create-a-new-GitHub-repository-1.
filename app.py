
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Upload transaction data to identify potentially fraudulent transactions.")

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

try:
    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.subheader("Transaction Data")
        st.dataframe(data.head())

        predictions = model.predict(data)

        results = data.copy()
        results["Prediction"] = [
            "Fraud" if value == 1 else "Legitimate"
            for value in predictions
        ]

        st.subheader("Prediction Results")
        st.dataframe(results)

        fraud_count = (predictions == 1).sum()
        st.metric("Potential Fraud Transactions", int(fraud_count))

except FileNotFoundError:
    st.warning(
        "Model not found. Run train.py first to create the trained model."
    )
