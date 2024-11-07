from pydantic import BaseModel, EmailStr, HttpUrl

class NLPDataInput(BaseModel):
    text: list[str]
    user_id: EmailStr

class ImageDataInput(BaseModel):
    url: list[HttpUrl]
    user_id: EmailStr

class NLPDataOutput(BaseModel):
    model_name: str
    text: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: int

class ImageDataOutput(BaseModel):
    model_name: str
    url: list[HttpUrl]
    labels: list[str]
    scores: list[float]
    prediction_time: int

class LinearDataInput(BaseModel):
    x1: int
    x2: int

class LinearDataOutput(BaseModel):
    data: list[int]
    prediction: float
    prediction_time: int