import streamlit as st
import joblib

model = joblib.load("spam-detection.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("📧 Spam vs Ham Classifier")
st.write("Enter a message to check if it's spam or ham.")

user_input = st.text_area("Message:", placeholder="Type your email or SMS message here...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        prediction_proba = model.predict_proba(transformed)[0]

        if prediction == 0:
            st.error(f"🚫 Prediction: **Spam** ({prediction_proba[0]*100:.2f}% confidence)")
        else:
            st.success(f"✅ Prediction: **Ham** ({prediction_proba[1]*100:.2f}% confidence)")
