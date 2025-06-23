from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_fake_news

# Create app project
app = FastAPI(title= "Fake News Detection")

# Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def root():
    return {"message": "Fake News Detection API is running"}

class TextRequest(BaseModel):
    text: str
    
@app.post("/predict")
def predict_news(request: TextRequest):
    label = predict_fake_news(request.text)
    return {"prediction": label}
