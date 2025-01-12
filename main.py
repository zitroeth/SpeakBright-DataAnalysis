from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np

origins = [
    "http://localhost:5173",
]

# Define input and output models
class DataPoint(BaseModel):
    timeTakenIndependence: float  # The independence time in milliseconds

class PredictionRequest(BaseModel):
    data: List[DataPoint]
    start: int  # Start index for prediction
    end: int    # End index for prediction

class PredictionResponse(BaseModel):
    predictedSum: float

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Run with `uvicorn filename:app --reload` (adjust filename accordingly)
# fastapi dev main.py --port 5174