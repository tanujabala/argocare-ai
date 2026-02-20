import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Argocare AI", layout="wide")

# ----------------------
# LOAD MODEL
# ----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_best.keras")

model = load_model()

# ----------------------
# CLASS NAMES (PUT YOUR 38)
# ----------------------
class_names = [
    "Apple___Scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    # ADD ALL 38 CLASSES HERE
]

# ----------------------
# DISEASE INFO (Add more later)
# ----------------------
disease_info = {
    "Apple___Scab": {
        "description": "Fungal disease causing dark lesions on leaves.",
        "treatment": "Use fungicides and remove infected leaves."
    },
    "Apple___Black_rot": {
        "description": "Causes fruit rot and leaf spots.",
        "treatment": "Prune infected branches and apply fungicide."
    }
}

# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.title("🌿 Argocare AI")
page = st.sidebar.radio("Navigation", ["Home", "Detect Disease", "About"])

# ----------------------
# HOME PAGE
# ----------------------
if page == "Home":
    st.title("AI Powered Plant Disease Detection")
    st.write("Upload a leaf image and get instant AI-based diagnosis.")
    st.success("Built using MobileNetV2 + Transfer Learning")

# ----------------------
# DETECTION PAGE
# ----------------------
elif page == "Detect Disease":

    st.header("Upload Plant Leaf Image")

    uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        disease_name = class_names[predicted_class]

        st.subheader("Prediction Result")
        st.success(f"{disease_name}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Show probability chart
        st.subheader("Confidence Distribution")
        df = pd.DataFrame(prediction[0], index=class_names, columns=["Probability"])
        df = df.sort_values("Probability", ascending=False).head(5)
        st.bar_chart(df)

        # Show disease info
        if disease_name in disease_info:
            st.subheader("Disease Description")
            st.write(disease_info[disease_name]["description"])

            st.subheader("Recommended Treatment")
            st.write(disease_info[disease_name]["treatment"])

        # Download report
        report = f"""
        Argocare AI Diagnosis Report

        Disease: {disease_name}
        Confidence: {confidence:.2f}%
        """

        st.download_button(
            label="Download Report",
            data=report,
            file_name="diagnosis_report.txt",
            mime="text/plain"
        )

# ----------------------
# ABOUT PAGE
# ----------------------
elif page == "About":
    st.title("About Argocare AI")
    st.write("""
    Argocare AI is an AI-powered plant disease detection system 
    built using Transfer Learning with MobileNetV2.
    It classifies plant diseases across 38 categories.
    """)

    st.write("Developed for Hackathon Presentation 🚀")