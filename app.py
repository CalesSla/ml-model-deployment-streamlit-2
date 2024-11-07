from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput, LinearDataInput, LinearDataOutput
from scripts import s3
from fastapi import FastAPI, Request
import uvicorn
import os
import pickle
import numpy as np

import time
import torch
from transformers import pipeline, AutoImageProcessor
import warnings
warnings.filterwarnings("ignore")

model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

force_download = False

model_name = "tinybert-sentiment-analysis/"
local_path = "ml-models/" + model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
sentiment_model = pipeline("text-classification", model=local_path, device=device)

model_name = "tinybert-disaster-tweet/"
local_path = "ml-models/" + model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
twitter_model = pipeline("text-classification", model=local_path, device=device)


model_name = 'vit-human-pose-classification/'
local_path = 'ml-models/' + model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
pose_model = pipeline('image-classification', model=local_path, device=device, image_processor=image_processor)


model_name = "LinearModel/"
local_path = "ml-models/" + model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
with open(local_path + "linear_regression_model.pkl", 'rb') as f:
    linear_model = pickle.load(f)



@app.get("/")
def read_root():
    return "Hello! I am up!"

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    predictiontime = int((end - start)*1000)
    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    output = NLPDataOutput(model_name="tinybert-sentiment-analysis",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=predictiontime)

    return output

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = twitter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)
    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    return output

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    start = time.time()
    output = pose_model(str(data.url[0]))
    end = time.time()
    prediction_time = int((end-start)*1000)
    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    output = ImageDataOutput(model_name="vit-human-pose-classification",
                             url=data.url,
                             labels=labels,
                             scores=scores,
                             prediction_time=prediction_time)

    return output

@app.post("/api/v1/linreg")
def linreg(data: LinearDataInput):
    values = [[data.x1, data.x2]]
    start = time.time()
    output = linear_model.predict(values)
    end = time.time()
    prediction_time = int((end-start)*1000)
    output = LinearDataOutput(data=values[0],
                              prediction=output.tolist()[0],
                              prediction_time=prediction_time)
    return output


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)