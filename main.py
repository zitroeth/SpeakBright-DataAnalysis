from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import requests
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import hashlib

# Load environment variables from .env file
load_dotenv()

origins = [
    "http://localhost:5173",
]

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input and output models
class DataPoint(BaseModel):
    timeTakenIndependence: float  # The independence time in milliseconds

class PredictionRequest(BaseModel):
    data: List[DataPoint]
    start: int  # Start index for prediction
    end: int    # End index for prediction

class PredictionResponse(BaseModel):
    predictedSum: float

@app.post("/simple-exponential-smoothing/", response_model=PredictionResponse)
def simple_exponential_smoothing(request: PredictionRequest):
    try:
        # Extract timeTakenIndependence values
        values = [item.timeTakenIndependence for item in request.data]

        # Check for sufficient data points
        if len(values) < 2:
            return PredictionResponse(predictedSum=0)

        # Fit the simple exponential smoothing model
        model = SimpleExpSmoothing(np.array(values))
        
        # Use least_squares approximation to get smoothing constant
        fit = model.fit(smoothing_level=None, method='least_squares', optimized=True)  
        
        # Predict the future values
        forecast = fit.predict(start=request.start, end=request.end - 1)

        # Sum the predicted values in the specified range
        predicted_sum = sum(forecast)

        return PredictionResponse(predictedSum=predicted_sum)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EmailTemplate(BaseModel):
    from_email: str
    to_email: str
    template_alias: str
    template_model: dict

@app.post("/send-email-with-template/")
def send_email_with_template(email_template: EmailTemplate):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Postmark-Server-Token": os.getenv("POSTMARK_API_TOKEN"),
    }
    data = {
        "From": email_template.from_email,
        "To": email_template.to_email,
        "TemplateAlias": email_template.template_alias,
        "TemplateModel": email_template.template_model
    }
    try:
        response = requests.post(os.getenv("POSTMARK_URL"), headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=str(http_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    
    return response.json()

class EncodeRequest(BaseModel):
    email: str
    type: str

class EncodeResponse(BaseModel):
    encoded: str
    
@app.post("/encode", response_model=EncodeResponse)
def encode_string(request: EncodeRequest):
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=500, detail="Secret key not configured.")
    
    # Build the base string using only the email and secret, plus the provided type.
    base_string = f"{request.email}-{request.type}-{secret}"
    
    # Compute the SHA-256 hash of the base string.
    hash_hex = hashlib.sha256(base_string.encode("utf-8")).hexdigest()
    
    return {"encoded": hash_hex}

# Run with `uvicorn filename:app --reload` (adjust filename accordingly)
# fastapi dev main.py --port 5174