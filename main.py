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
import math

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

# Extend PredictionResponse to carry smoothing level
class PredictionResponse(BaseModel):
    predictedSum: float
    smoothingLevel: float

@app.post("/simple-exponential-smoothing/", response_model=PredictionResponse)
def simple_exponential_smoothing(request: PredictionRequest):
    try:
        # Extract timeTakenIndependence values
        values = [item.timeTakenIndependence for item in request.data]

        # Check for sufficient data points
        if len(values) < 2:
            return PredictionResponse(predictedSum=0, smoothingLevel=0)

        # Fit the simple exponential smoothing model
        model = SimpleExpSmoothing(np.array(values))
        
        # Use least_squares approximation to get smoothing constant
        fit = model.fit(smoothing_level=None, method='least_squares', optimized=True)  
        smoothing_level = fit.params["smoothing_level"]
        print(f"Fitted smoothing level: {smoothing_level}\nCount: {len(values)}")
        
        # Predict the future values
        forecast = fit.predict(start=request.start, end=request.end - 1)

        # Sum the predicted values in the specified range
        predicted_sum = sum(forecast)

        return PredictionResponse(
            predictedSum=predicted_sum,
            smoothingLevel=smoothing_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    query = codes_ref \
            .where(field_path="receiver_email", op_string="==", value=receiver_email) \
            .where(field_path="requester_email", op_string="==", value=requester_email) \
            .where(field_path="code", op_string="==", value=code) \
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
    user_docs = list(db.collection("users") \
                     .where(field_path="email", op_string="==", value=receiver_email) \
                     .limit(1).stream())
    if not user_docs:
        # Optionally, handle case when no user document is found.
        return
    user_doc_ref = user_docs[0]
    user_doc = user_doc_ref.to_dict()
    user_doc_id = user_doc_ref.id

    if type == "guardian-monitoring-request":
        # For guardian monitor request, fetch the parent document in the "user_admin" collection where email equals requester_email.
        admin_docs = list(db.collection("user_admin") \
                          .where(field_path="email", op_string="==", value=requester_email) \
                          .limit(1).stream())
        if not admin_docs:
            # Optionally, create a new parent document if one doesn't exist.
            # Here we assume a document should already exist. Adjust as needed.
            raise Exception("Parent document for requester not found in user_admin collection.")
        admin_doc_ref = admin_docs[0].reference
        # Add a new document (with the same id as the user) to the "guardians" subcollection.
        admin_doc_ref.collection("guardians").document(user_doc_id).set({**user_doc, "id": user_doc_id})
    elif type == "student-monitoring-request":
        # For student monitor request, fetch the parent document in the "user_guardian" collection where email equals requester_email.
        guardian_docs = list(db.collection("user_guardian") \
                             .where(field_path="email", op_string="==", value=requester_email) \
                             .limit(1).stream())
        if not guardian_docs:
            raise Exception("Parent document for requester not found in user_guardian collection.")
        guardian_doc_ref = guardian_docs[0].reference
        # Add a new document to the "students" subcollection.
        guardian_doc_ref.collection("students").document(user_doc_id).set({**user_doc, "id": user_doc_id})

# Models for EMA mobile endpoint
class PhasePrediction(BaseModel):
    phase: int
    predictedMili: float
    predictedString: str

def convert_milliseconds_to_readable_string(milliseconds: float, shorten: bool = False) -> str:
    seconds = int((milliseconds / 1000) % 60)
    minutes = int((milliseconds / (1000 * 60)) % 60)
    hours   = int((milliseconds / (1000 * 60 * 60)) % 24)
    days    = int((milliseconds / (1000 * 60 * 60 * 24)) % 30.44)
    months  = int((milliseconds / (1000 * 60 * 60 * 24 * 30.44)) % 12)
    years   = int(milliseconds / (1000 * 60 * 60 * 24 * 365.25))

    parts: List[str] = []

    def add(unit_val, name):
        if unit_val:
            parts.append(f"{unit_val} {name}" + ("s" if unit_val != 1 else ""))

    if shorten:
        # only top 3 non-zero units
        units = [("year", years), ("month", months), ("day", days),
                 ("hour", hours), ("minute", minutes), ("second", seconds)]
        count = 0
        for name, val in units:
            if val and count < 3:
                add(val, name)
                count += 1
    else:
        add(years,   "year")
        add(months,  "month")
        add(days,    "day")
        add(hours,   "hour")
        add(minutes, "minute")
        add(seconds, "second")
        if not parts:
            parts.append("0 seconds")

    return ", ".join(parts)


@app.get("/ema-mobile/{student_id}", response_model=List[PhasePrediction])
def ema_mobile(student_id: str):
    predictions: List[PhasePrediction] = []

    for phase in range(1, 4):
        # 1) build map of earliest “started at” per card from activity_log
        started_at: dict = {}
        phase_col = db.collection("activity_log").document(student_id)\
                      .collection("phase").document(str(phase)).collection("session")
        for sess in phase_col.stream():
            sess_ts = sess.to_dict().get("timestamp")
            if not sess_ts:
                continue
            for trial in sess.reference.collection("trialPrompt").stream():
                d = trial.to_dict()
                cid, ts = d.get("cardID"), d.get("timestamp")
                if not cid or not ts:
                    continue
                prev = started_at.get(cid)
                if not prev or ts < prev:
                    started_at[cid] = ts

        # 2) gather all cards for this phase and count total
        phase_cards = []
        for card in db.collection("cards") \
                      .where(field_path="userId", op_string="==", value=student_id) \
                      .stream():
            cd = card.to_dict()
            cat = cd.get("category", "")
            if phase == 1 \
               or (phase == 2 and cat != "Emotions") \
               or (phase == 3 and cat == "Emotions"):
                phase_cards.append(card)
        total_cards = len(phase_cards)

        # 3) compute durations only for those that are independent and have completed
        completion_list = []
        for card in phase_cards:
            cd = card.to_dict()
            if not cd.get(f"phase{phase}_independence", False):
                continue
            comp_ts = cd.get(f"phase{phase}_completion", False)
            if not comp_ts:
                continue
            start_ts = started_at.get(card.id)
            if not start_ts:
                continue
            dur = (comp_ts - start_ts).total_seconds() * 1000
            title = cd.get("title", "")
            completion_list.append((comp_ts, dur, card.id, title, start_ts))

        # 4) sort & extract durations
        completion_list.sort(key=lambda x: x[0])
        durations = [dur for _, dur, _, _, _ in completion_list]
        if not durations:
            continue

        # 5) forecast from next index up to total_cards
        req = PredictionRequest(
            data=[DataPoint(timeTakenIndependence=d) for d in durations],
            start=len(durations),
            end=total_cards
        )
        resp = simple_exponential_smoothing(req)
        predictions.append(
            PhasePrediction(
                phase=phase,
                predictedMili=resp.predictedSum,
                predictedString=convert_milliseconds_to_readable_string(
                    resp.predictedSum, shorten=True
                )
            )
        )

    # print formatted strings before returning
    for p in predictions:
        print(f"Phase {p.phase} prediction: {p.predictedString}")

    return predictions

# Models for raw output
class CardDetail(BaseModel):
    cardId: str
    title: str
    startedAt: str
    completedAt: str
    durationTookForIndependence: float
    durationTookForIndependenceString: str  # new field

class PhaseDetail(BaseModel):
    phase: int
    phasePrediction: float
    smoothingLevel: float
    phasePredictionString: str
    nonIndependentCount: int             # new field
    cards: List[CardDetail]

@app.get("/ema-mobile-raw/{student_id}", response_model=List[PhaseDetail])
def ema_mobile_raw(student_id: str):
    raw_output: List[PhaseDetail] = []

    for phase in range(1, 4):
        # 1) build started_at map (same as ema_mobile)
        started_at: dict = {}
        phase_col = db.collection("activity_log")\
                      .document(student_id)\
                      .collection("phase")\
                      .document(str(phase))\
                      .collection("session")
        for sess in phase_col.stream():
            sess_ts = sess.to_dict().get("timestamp")
            if not sess_ts:
                continue
            for trial in sess.reference.collection("trialPrompt").stream():
                d = trial.to_dict()
                cid, ts = d.get("cardID"), d.get("timestamp")
                if not cid or not ts:
                    continue
                prev = started_at.get(cid)
                if not prev or ts < prev:
                    started_at[cid] = ts

        # 2) gather and filter cards
        phase_cards = []
        for card in db.collection("cards")\
                      .where(field_path="userId", op_string="==", value=student_id)\
                      .stream():
            cd = card.to_dict()
            cat = cd.get("category", "")
            if phase == 1 \
               or (phase == 2 and cat != "Emotions") \
               or (phase == 3 and cat == "Emotions"):
                phase_cards.append(card)
        total_cards = len(phase_cards)

        # 3) compute durations for independent cards
        completion_list = []
        for card in phase_cards:
            cd = card.to_dict()
            if not cd.get(f"phase{phase}_independence", False):
                continue
            comp_ts = cd.get(f"phase{phase}_completion")
            if not comp_ts:
                continue
            start_ts = started_at.get(card.id)
            if not start_ts:
                continue
            dur = (comp_ts - start_ts).total_seconds() * 1000
            title = cd.get("title", "")
            completion_list.append((comp_ts, dur, card.id, title, start_ts))

        if not completion_list:
            continue

        # 4) sort & forecast
        completion_list.sort(key=lambda x: x[0])
        durations = [dur for _, dur, _, _, _ in completion_list]

        req = PredictionRequest(
            data=[DataPoint(timeTakenIndependence=d) for d in durations],
            start=len(durations),
            end=total_cards
        )
        resp = simple_exponential_smoothing(req)

        # 5) build card details including the string versions
        cards = [
            CardDetail(
                cardId=cid,
                title=title,
                startedAt=start_ts.isoformat(),
                completedAt=comp_ts.isoformat(),
                durationTookForIndependence=dur,
                durationTookForIndependenceString=convert_milliseconds_to_readable_string(
                    dur, shorten=True
                )
            )
            for comp_ts, dur, cid, title, start_ts in completion_list
        ]

        non_independent = total_cards - len(completion_list)

        phase_output = PhaseDetail(
            phase=phase,
            phasePrediction=resp.predictedSum,
            smoothingLevel=resp.smoothingLevel,
            phasePredictionString=convert_milliseconds_to_readable_string(
                resp.predictedSum, shorten=True
            ),
            nonIndependentCount=non_independent,
            cards=cards
        )

        print(phase_output.dict())
        raw_output.append(phase_output)

    return raw_output

# Run with `uvicorn filename:app --reload` (adjust filename accordingly)
# fastapi dev main.py --port 5174

if __name__ == "__main__":
    import uvicorn
    # read PORT from environment (default to 4000)
    port = int(os.getenv("PORT", 4000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )