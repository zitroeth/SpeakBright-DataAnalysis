from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import requests
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import json
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
from dateutil import parser 

# Load environment variables from .env file
load_dotenv()

# Use Firebase Admin SDK
cred = credentials.Certificate('./ServiceAccountKey.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()

# Retrieve the secret key from the environment variable
key_str = os.getenv("SECRET_KEY")
if not key_str:
    raise Exception("SECRET_KEY environment variable is not configured.")
# Create Fernet instance using the encoded key
f = Fernet(key_str.encode())

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
        print(f"Fitted smoothing level: {fit.params['smoothing_level']}\nCount: {len(values)}")
        # Predict the future values
        forecast = fit.predict(start=request.start, end=request.end - 1)

        # Sum the predicted values in the specified range
        predicted_sum = sum(forecast)

        return PredictionResponse(predictedSum=predicted_sum)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# class EmailTemplate(BaseModel):
#     from_email: str
#     to_email: str
#     template_alias: str
#     template_model: dict

# @app.post("/send-email-with-template/")
# def send_email_with_template(email_template: EmailTemplate):
#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#         "X-Postmark-Server-Token": os.getenv("POSTMARK_API_TOKEN"),
#     }
#     data = {
#         "From": email_template.from_email,
#         "To": email_template.to_email,
#         "TemplateAlias": email_template.template_alias,
#         "TemplateModel": email_template.template_model
#     }
#     try:
#         response = requests.post(os.getenv("POSTMARK_URL"), headers=headers, json=data)
#         response.raise_for_status()
#     except requests.exceptions.HTTPError as http_err:
#         raise HTTPException(status_code=response.status_code, detail=str(http_err))
#     except Exception as err:
#         raise HTTPException(status_code=500, detail=str(err))
    
#     return response.json()

# class EncodeRequest(BaseModel):
#     email: str
#     type: str

# class EncodeResponse(BaseModel):
#     encoded: str
    
# @app.post("/encode", response_model=EncodeResponse)
# def encode_json(request: EncodeRequest):
#     try:
#         # Convert JSON object to string
#         json_str = json.dumps(request.data)
#         # Encrypt the string (must encode to bytes)
#         encrypted_bytes = f.encrypt(json_str.encode())
#         # Return the encrypted string (decoded for JSON compatibility)
#         return {"encoded": encrypted_bytes.decode("utf-8")}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Encoding failed")

# # Models for decode endpoint
# class DecodeRequest(BaseModel):
#     encoded: str

# class DecodeResponse(BaseModel):
#     data: dict

# @app.post("/decode", response_model=DecodeResponse)
# def decode_json(request: DecodeRequest):
#     try:
#         # Decrypt the encoded string (after encoding it back to bytes)
#         decrypted_bytes = f.decrypt(request.encoded.encode("utf-8"))
#         # Convert decrypted bytes back to a string
#         json_str = decrypted_bytes.decode("utf-8")
#         # Convert JSON string to JSON object
#         data = json.loads(json_str)
#         return {"data": data}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Decoding failed")

# Email templating model and send function.
# Email templating model and send function.
class EmailTemplate(BaseModel):
    from_email: str
    to_email: str
    template_alias: str
    template_model: dict

def send_email(email_template: EmailTemplate) -> dict:
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

# Request model for invite creation.
class CreateInviteRequest(BaseModel):
    receiver_email: str
    requester_email: str
    requester_name: str
    type: str

# Response model for invite creation.
class CreateInviteResponse(BaseModel):
    message: str
    verification_link: str

# Endpoint to create an invite.
@app.post("/create-invite", response_model=CreateInviteResponse)
def create_invite(invite: CreateInviteRequest):
    # Prepare JSON data that includes only email and type.
    payload = {"receiver_email": invite.receiver_email, "requester_email": invite.requester_email, "type": invite.type}
    json_str = json.dumps(payload)
    
    # Encrypt the JSON string into a code.
    code = f.encrypt(json_str.encode("utf-8")).decode("utf-8")
    
    # Record the creation time (UTC).
    created_at = datetime.datetime.now(datetime.timezone.utc)

    # Store the invite into Firestore collection "codes".
    doc_data = {
        "receiver_email": invite.receiver_email,
        "requester_email": invite.requester_email,
        "type": invite.type,
        "code": code,
        "created_at": created_at
    }
    db.collection("codes").add(doc_data)

    # Send the email using the email templating API.
    from_email = os.getenv("FROM_EMAIL")
    template_alias = invite.type
    # Build the accept_link with query parameters.
    accept_link = (
        f"{os.getenv('BACKEND_URL')}verify-invite?"
        f"receiver_email={invite.receiver_email}&requester_email={invite.requester_email}&type={invite.type}&code={code}"
        f"&created_at={created_at.isoformat()}"
    )
    template_model = {
        "product_name": "SpeakBright",
        "product_url": os.getenv("FRONTEND_URL"),
        "company_name": "SpeakBright",
        "company_address": "speakbright@speakbright.com",
        "receiver_name": "",
        "requester_name": invite.requester_name,
        "requester_email": invite.requester_email,
        "accept_link": accept_link,
        "decline_link": f"{os.getenv('BACKEND_URL')}decline-invite",
    }
    email_template = EmailTemplate(
        from_email=from_email,
        to_email=invite.receiver_email,
        template_alias=template_alias,
        template_model=template_model
    )
    # Call the function to send email.
    send_email(email_template)

    return CreateInviteResponse(
        message="Invite created successfully and email sent.",
        verification_link=accept_link
    )

# Endpoint to verify the invite.
@app.get("/verify-invite")
def verify_invite(
    receiver_email: str = Query(...),
    requester_email: str = Query(...),
    type: str = Query(...),
    code: str = Query(...),
    created_at: str = Query(...)
):
    # Validate created_at using dateutil.parser which supports ISO 8601.
    try:
        created_at_dt = parser.parse(created_at)
        # Ensure the datetime is offset-aware.
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid created_at format")
    
    # Check if the invitation has not expired (24-hour validity).
    now = datetime.datetime.now(datetime.timezone.utc)
    if now - created_at_dt > datetime.timedelta(hours=24):
        raise HTTPException(status_code=400, detail="Verification link expired")
    
    # Decrypt the code and verify that it matches email and type.
    try:
        decrypted_bytes = f.decrypt(code.encode("utf-8"))
        payload = json.loads(decrypted_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid verification code")
    
    if payload.get("receiver_email") != receiver_email or payload.get("requester_email") != requester_email or payload.get("type") != type:
        raise HTTPException(status_code=400, detail="Verification data mismatch")
    
    # Query Firestore for a matching invitation document.
    codes_ref = db.collection("codes")
    query = codes_ref.where("receiver_email", "==", receiver_email)\
                     .where("requester_email", "==", requester_email)\
                     .where("code", "==", code)\
                     .limit(1).stream()
    
    doc_found = None
    for doc in query:
        doc_found = doc
        break
    if not doc_found:
        raise HTTPException(status_code=400, detail="No matching invite found or already verified")
    
    # Delete the invitation document if the verification is successful.
    doc_found.reference.delete()

    addToMonitorList(receiver_email, requester_email, type)
    return {"message": "Invite verified successfully"}

def addToMonitorList(receiver_email: str, requester_email: str, type: str):
    # Fetch the user document from "users" collection where field email equals receiver_email.
    user_docs = list(db.collection("users").where("email", "==", receiver_email).limit(1).stream())
    if not user_docs:
        # Optionally, handle case when no user document is found.
        return
    user_doc_ref = user_docs[0]
    user_doc = user_doc_ref.to_dict()
    user_doc_id = user_doc_ref.id

    if type == "guardian-monitoring-request":
        # For guardian monitor request, fetch the parent document in the "user_admin" collection where email equals requester_email.
        admin_docs = list(db.collection("user_admin").where("email", "==", requester_email).limit(1).stream())
        if not admin_docs:
            # Optionally, create a new parent document if one doesn't exist.
            # Here we assume a document should already exist. Adjust as needed.
            raise Exception("Parent document for requester not found in user_admin collection.")
        admin_doc_ref = admin_docs[0].reference
        # Add a new document (with the same id as the user) to the "guardians" subcollection.
        admin_doc_ref.collection("guardians").document(user_doc_id).set({**user_doc, "id": user_doc_id})
    elif type == "student-monitoring-request":
        # For student monitor request, fetch the parent document in the "user_guardian" collection where email equals requester_email.
        guardian_docs = list(db.collection("user_guardian").where("email", "==", requester_email).limit(1).stream())
        if not guardian_docs:
            raise Exception("Parent document for requester not found in user_guardian collection.")
        guardian_doc_ref = guardian_docs[0].reference
        # Add a new document to the "students" subcollection.
        guardian_doc_ref.collection("students").document(user_doc_id).set({**user_doc, "id": user_doc_id})
# Run with `uvicorn filename:app --reload` (adjust filename accordingly)
# fastapi dev main.py --port 5174