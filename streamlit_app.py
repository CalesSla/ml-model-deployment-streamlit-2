import streamlit as st
import requests
from scripts import s3

API_URL = "http://127.0.0.1:8000/api/v1/"
headers = {
    "Content-Type": "application/json"
}

st.title("ML Model Serving Over REST API")

model = st.selectbox("Select Model",
                     ["Sentiment Classifier", "Disaster Classifier", "Pose Classifier", "Linear Regressor"])

if model == "Sentiment Classifier":
    text = st.text_area("Enter Your Movie Review")
    user_id = st.text_input("Enter user id", "udemy@slava.com")
    
    data = {"text": [text], "user_id": user_id}
    model_api = "sentiment_analysis"

elif model == "Disaster Classifier":
    text = st.text_area("Enter Your Twitter text")
    user_id = st.text_input("Enter user id", "udemy@slava.com")
    data = {"text": [text], "user_id": user_id}
    model_api = "disaster_classifier"

elif model == "Pose Classifier":
    select_file = st.radio("Select the image source", ["Local", "URL"])

    if select_file == "URL":
        url = st.text_input("Enter your image URL")

    else:
        image = st.file_uploader("Upload the image", type=["jgp", "jpeg", "png"])
        file_name = "images/temp.jpg"
        if image is not None:
            with open(file_name, "wb") as f:
                f.write(image.read())
        url = s3.upload_image_to_s3(file_name)
    
    user_id = st.text_input("Enter user id", "udemy@slava.com")
    data = {"url": [url], "user_id": user_id}
    model_api = "pose_classifier"

elif model == "Linear Regressor":
    x1 = st.number_input("x1",
                         value=10,
                         step=1,
                         format="%d")
    x2 = st.number_input("x2",
                         value=10,
                         step=1,
                         format="%d")
    data = {"x1": x1, "x2": x2}
    model_api = "linreg"

if st.button("Predict"):
    with st.spinner("Predicting... Please Wait!!!"):
        response = requests.post(API_URL + model_api, headers=headers, json=data)
        output = response.json()
    
    st.write(output)
