import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import requests
from streamlit_lottie import st_lottie
import plotly.express as px

# Load the pre-trained ResNet50 model (cached for efficiency)
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()

def predict_and_show(img_data, confidence_threshold):
    img = Image.open(img_data)
    img = img.resize((224, 224), Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Filter predictions based on confidence_threshold
    filtered_predictions = [pred for pred in decoded_predictions if pred[2]*100 >= confidence_threshold]
    return filtered_predictions

# Sidebar - Advanced options
st.sidebar.header('Advanced Options')
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0, 100, 20)


# **Real-Time Classification UI**
st.set_page_config(page_title="Image Classifier", page_icon="\U0001F3A8", layout="centered")
st.title('\U0001F3A8  Image Classifier')


# **Drag & Drop / Multiple Image Upload**
uploaded_files = st.file_uploader(
    "Drag and drop image(s) here or browse...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# **REAL-TIME CLASSIFICATION**
if uploaded_files:
    for uploaded_file in uploaded_files:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Display loading animation
        lottie_url = "https://lottie.host/48a98916-5ce1-41f7-a043-b5932bc5c542/w183dqaRuZ.json"
        lottie_animation = load_lottieurl(lottie_url)
        st_lottie(lottie_animation, height=200, key=str(uploaded_file))

        # Run Prediction Instantly (Real-Time)
        predictions = predict_and_show(uploaded_file, confidence_threshold)

        st.title("Results")
        if predictions:
            labels = [pred[1] for pred in predictions]
            scores = [pred[2] * 100 for pred in predictions]
            fig = px.bar(x=labels, y=scores, labels={'x':'Predicted Class', 'y':'Confidence (%)'}, title="Top Predictions")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("See Prediction Details"):
                for pred in predictions:
                    st.write(f"✅ **{pred[1].capitalize()}**: {pred[2]*100:.2f}% Confidence Level")
        else:
            st.write("⚠️ No Predictions with Confidence Level above the Threshold.")

st.markdown("---")
