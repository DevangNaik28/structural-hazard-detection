import os
from datetime import datetime
from pathlib import Path
import pytz
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from twilio.twiml.messaging_response import MessagingResponse

from bridges.synthetic_generator import generate_sensor_data as bridge_sensor
from buildings.synthetic_generator import generate_sensor_data as building_sensor
from dams.synthetic_generator import generate_sensor_data as dam_sensor
from tunnels.synthetic_generator import generate_sensor_data as tunnel_sensor
from highways.synthetic_generator import generate_sensor_data as highway_sensor

from shared.pipeline import (
    load_models,
    predict_bridge_risk,
    predict_building_risk,
    predict_dam_risk,
    predict_tunnel_risk,
    predict_highway_risk
)

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Infra Mind Vision API",
    description="AI-powered structural health monitoring for bridges, buildings, dams, tunnels and highways",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Escalation Status Tracking ────────────────────────────────────────────────

escalation_status = {}
number_to_ticket = {}

# ── Load All Models at Startup ────────────────────────────────────────────────

MODELS = {}

@app.on_event("startup")
def load_all_models():
    global MODELS
    MODELS = {
        "bridges":   load_models("models/bridges/isolation_forest.pkl",   "models/bridges/risk_model.pkl"),
        "buildings": load_models("models/buildings/isolation_forest.pkl", "models/buildings/risk_model.pkl"),
        "dams":      load_models("models/dams/isolation_forest.pkl",      "models/dams/risk_model.pkl"),
        "tunnels":   load_models("models/tunnels/isolation_forest.pkl",   "models/tunnels/risk_model.pkl"),
        "highways":  load_models("models/highways/isolation_forest.pkl",  "models/highways/risk_model.pkl"),
    }
    print("✅ All models loaded successfully")


# ── Structure Registry ────────────────────────────────────────────────────────

STRUCTURES = {
    "bridges": {
        "worli_sealink":   {"bridge_id": "worli_sealink",   "age_of_bridge": 23,  "last_maintenance_days": 45,  "stress_level": 0},
        "bandra_worli":    {"bridge_id": "bandra_worli",    "age_of_bridge": 15,  "last_maintenance_days": 90,  "stress_level": 0},
        "pamban_bridge":   {"bridge_id": "pamban_bridge",   "age_of_bridge": 110, "last_maintenance_days": 420, "stress_level": 2},
        "howrah_bridge":   {"bridge_id": "howrah_bridge",   "age_of_bridge": 80,  "last_maintenance_days": 300, "stress_level": 2},
        "vidyasagar_setu": {"bridge_id": "vidyasagar_setu", "age_of_bridge": 28,  "last_maintenance_days": 150, "stress_level": 1},
    },
    "buildings": {
        "csmt":            {"building_id": "csmt",            "building_age_years": 130, "floor_count": 4,  "occupancy": 300, "stress_level": 2},
        "bkc_tower":       {"building_id": "bkc_tower",       "building_age_years": 10,  "floor_count": 32, "occupancy": 800, "stress_level": 0},
        "mantralaya":      {"building_id": "mantralaya",      "building_age_years": 60,  "floor_count": 6,  "occupancy": 500, "stress_level": 1},
        "nariman_point":   {"building_id": "nariman_point",   "building_age_years": 45,  "floor_count": 20, "occupancy": 600, "stress_level": 1},
        "atal_setu_plaza": {"building_id": "atal_setu_plaza", "building_age_years": 5,   "floor_count": 15, "occupancy": 400, "stress_level": 0},
    },
    "dams": {
        "koyna_dam":       {"dam_id": "koyna_dam",       "dam_type": "concrete", "dam_height_m": 103, "dam_age_years": 62, "stress_level": 1},
        "bhakra_dam":      {"dam_id": "bhakra_dam",      "dam_type": "concrete", "dam_height_m": 226, "dam_age_years": 65, "stress_level": 2},
        "hirakud_dam":     {"dam_id": "hirakud_dam",     "dam_type": "earthen",  "dam_height_m": 61,  "dam_age_years": 70, "stress_level": 2},
        "tehri_dam":       {"dam_id": "tehri_dam",       "dam_type": "earthen",  "dam_height_m": 260, "dam_age_years": 20, "stress_level": 0},
        "nagarjuna_sagar": {"dam_id": "nagarjuna_sagar", "dam_type": "concrete", "dam_height_m": 124, "dam_age_years": 58, "stress_level": 1},
    },
    "tunnels": {
        "mumbai_metro":     {"tunnel_id": "mumbai_metro",     "tunnel_type": "metro", "tunnel_length_m": 3200, "tunnel_age_years": 8,  "traffic_volume": 1200, "stress_level": 0},
        "bandra_tunnel":    {"tunnel_id": "bandra_tunnel",    "tunnel_type": "road",  "tunnel_length_m": 800,  "tunnel_age_years": 30, "traffic_volume": 900,  "stress_level": 1},
        "rohtang_tunnel":   {"tunnel_id": "rohtang_tunnel",   "tunnel_type": "road",  "tunnel_length_m": 9200, "tunnel_age_years": 5,  "traffic_volume": 600,  "stress_level": 0},
        "chenani_nashri":   {"tunnel_id": "chenani_nashri",   "tunnel_type": "road",  "tunnel_length_m": 9200, "tunnel_age_years": 8,  "traffic_volume": 700,  "stress_level": 1},
        "delhi_metro_blue": {"tunnel_id": "delhi_metro_blue", "tunnel_type": "metro", "tunnel_length_m": 5000, "tunnel_age_years": 20, "traffic_volume": 2000, "stress_level": 2},
    },
    "highways": {
        "mumbai_pune_expressway": {"highway_id": "mumbai_pune_expressway", "highway_type": "expressway", "highway_length_km": 94,   "highway_age_years": 24, "traffic_volume": 1800, "heavy_vehicle_count": 300, "stress_level": 1},
        "nh48_delhi_jaipur":      {"highway_id": "nh48_delhi_jaipur",      "highway_type": "national",   "highway_length_km": 270,  "highway_age_years": 30, "traffic_volume": 2000, "heavy_vehicle_count": 500, "stress_level": 2},
        "nh44_delhi_chennai":     {"highway_id": "nh44_delhi_chennai",     "highway_type": "national",   "highway_length_km": 3745, "highway_age_years": 40, "traffic_volume": 1500, "heavy_vehicle_count": 600, "stress_level": 2},
        "yamuna_expressway":      {"highway_id": "yamuna_expressway",      "highway_type": "expressway", "highway_length_km": 165,  "highway_age_years": 12, "traffic_volume": 1200, "heavy_vehicle_count": 200, "stress_level": 0},
        "eastern_peripheral":     {"highway_id": "eastern_peripheral",     "highway_type": "expressway", "highway_length_km": 135,  "highway_age_years": 6,  "traffic_volume": 900,  "heavy_vehicle_count": 150, "stress_level": 0},
    },
}

ANOMALY_TYPES = {
    "bridges":   {1: "vibration_spike",      2: "stress_concentration"},
    "buildings": {1: "stress_concentration", 2: "seismic_spike"},
    "dams":      {1: "seepage_anomaly",      2: "overtopping_risk"},
    "tunnels":   {1: "water_ingress",        2: "collapse_risk"},
    "highways":  {1: "overload_event",       2: "pavement_failure"},
}


# ── Sensor Spike Injection ────────────────────────────────────────────────────

def apply_stress(sensor_df, structure_type, stress_level):
    if stress_level == 0:
        return sensor_df
    df = sensor_df.copy()
    if structure_type == "bridges":
        if stress_level == 1:
            df["vibration"] *= 1.4; df["strain"] *= 1.2
        elif stress_level == 2:
            df["vibration"] *= 1.8; df["strain"] *= 1.5
            df["displacement"] *= 1.6; df["acceleration"] *= 1.5
    elif structure_type == "buildings":
        if stress_level == 1:
            df["vibration_x"] *= 1.4; df["strain"] *= 1.2; df["crack_width"] *= 1.3
        elif stress_level == 2:
            df["vibration_x"] *= 1.8; df["vibration_y"] *= 1.6
            df["strain"] *= 1.5; df["settlement"] += 0.8; df["crack_width"] *= 1.6
    elif structure_type == "dams":
        if stress_level == 1:
            df["seepage_rate"] *= 1.5; df["pore_water_pressure"] *= 1.3
        elif stress_level == 2:
            df["seepage_rate"] *= 1.9; df["pore_water_pressure"] *= 1.6
            df["crest_settlement"] += 0.8; df["crack_width"] *= 1.7
    elif structure_type == "tunnels":
        if stress_level == 1:
            df["water_ingress"] *= 1.5; df["crack_width"] *= 1.3; df["lining_pressure"] *= 1.2
        elif stress_level == 2:
            df["vibration"] *= 1.8; df["strain"] *= 1.5
            df["convergence"] *= 1.7; df["water_ingress"] *= 1.9
    elif structure_type == "highways":
        if stress_level == 1:
            df["pavement_strain"] *= 1.4; df["crack_density"] *= 1.3; df["axle_load"] *= 1.2
        elif stress_level == 2:
            df["pavement_strain"] *= 1.8; df["surface_deflection"] *= 1.6
            df["crack_density"] *= 1.9; df["rutting_depth"] *= 1.5
    return df


# ── Helper ────────────────────────────────────────────────────────────────────

def get_prediction(structure_type, structure_id):
    if structure_type not in STRUCTURES:
        raise HTTPException(status_code=404, detail=f"Structure type '{structure_type}' not found")
    if structure_id not in STRUCTURES[structure_type]:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")

    params = STRUCTURES[structure_type][structure_id]
    stress_level = params.get("stress_level", 0)
    sensor_params = {k: v for k, v in params.items() if k != "stress_level"}
    anomaly_model, risk_model = MODELS[structure_type]

    if structure_type == "bridges":
        sensor_df = bridge_sensor(**sensor_params)
        sensor_df = apply_stress(sensor_df, structure_type, stress_level)
        anomaly_flag, risk_score, risk_level = predict_bridge_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "buildings":
        sensor_df = building_sensor(**sensor_params)
        sensor_df = apply_stress(sensor_df, structure_type, stress_level)
        anomaly_flag, risk_score, risk_level = predict_building_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "dams":
        sensor_df = dam_sensor(**sensor_params)
        sensor_df = apply_stress(sensor_df, structure_type, stress_level)
        anomaly_flag, risk_score, risk_level = predict_dam_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "tunnels":
        sensor_df = tunnel_sensor(**sensor_params)
        sensor_df = apply_stress(sensor_df, structure_type, stress_level)
        anomaly_flag, risk_score, risk_level = predict_tunnel_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "highways":
        sensor_df = highway_sensor(**sensor_params)
        sensor_df = apply_stress(sensor_df, structure_type, stress_level)
        anomaly_flag, risk_score, risk_level = predict_highway_risk(sensor_df, anomaly_model, risk_model)

    anomaly_type = "none"
    if stress_level > 0:
        anomaly_type = ANOMALY_TYPES.get(structure_type, {}).get(stress_level, "anomaly_detected")

    return {
        "structure_id": structure_id,
        "structure_type": structure_type,
        "anomaly_detected": bool(anomaly_flag),
        "anomaly_type": anomaly_type,
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "status": "HIGH RISK" if anomaly_flag else "NORMAL",
        "sensor_data": sensor_df.to_dict(orient="records")[0]
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Infra Mind Vision API is running"}


@app.get("/api/health/{structure_type}/{structure_id}")
def get_structure_health(structure_type: str, structure_id: str):
    return get_prediction(structure_type, structure_id)


@app.get("/api/structures/{structure_type}")
def list_structures(structure_type: str):
    if structure_type not in STRUCTURES:
        raise HTTPException(status_code=404, detail=f"Structure type '{structure_type}' not found")
    return {"structure_type": structure_type, "structures": list(STRUCTURES[structure_type].keys())}


@app.get("/api/structures")
def list_all_structures():
    return {stype: list(structs.keys()) for stype, structs in STRUCTURES.items()}


class EscalateRequest(BaseModel):
    structure_id: str
    structure_type: str
    risk_level: str
    anomaly_description: str
    ticket_id: str


@app.get("/api/summary")
def get_all_summary():
    summary = []
    for structure_type, structures in STRUCTURES.items():
        for structure_id in structures:
            result = get_prediction(structure_type, structure_id)
            summary.append({
                "structure_id": result["structure_id"],
                "structure_type": result["structure_type"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "status": result["status"],
                "anomaly_detected": result["anomaly_detected"],
                "anomaly_type": result["anomaly_type"],
            })
    return {"total": len(summary), "structures": summary}


@app.post("/api/escalate")
def escalate_alert(request: EscalateRequest):
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        whatsapp_from = os.environ.get("TWILIO_WHATSAPP_FROM")
        whatsapp_to = os.environ.get("TWILIO_WHATSAPP_TO")

        if not all([account_sid, auth_token, whatsapp_from, whatsapp_to]):
            raise ValueError("Missing Twilio credentials in environment variables")

        client = Client(account_sid, auth_token)

        ist = pytz.timezone('Asia/Kolkata')
        ist_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')

        message_body = f"""
🚨 INFRAWATCH ALERT
Ticket: {request.ticket_id}
Structure: {request.structure_id} ({request.structure_type})
Risk Level: {request.risk_level}
Issue: {request.anomaly_description}
Time: {ist_time}
→ Immediate inspection required.
Reply Y to acknowledge this alert.
""".strip()

        message = client.messages.create(
            from_=whatsapp_from,
            body=message_body,
            to=whatsapp_to
        )

        escalation_status[request.ticket_id] = "PENDING"
        number_to_ticket[whatsapp_to] = request.ticket_id

        return {"success": True, "ticket_id": request.ticket_id, "message_sid": message.sid}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        form = await request.form()
        form_dict = dict(form)
        body = form_dict.get("Body", "").strip().upper()

        if body == "Y":
            if escalation_status:
                latest_ticket = list(escalation_status.keys())[-1]
                escalation_status[latest_ticket] = "ACKNOWLEDGED"

        resp = MessagingResponse()
        if body == "Y":
            resp.message("✅ Alert acknowledged. Inspection confirmed.")
        return PlainTextResponse(str(resp), media_type="application/xml")

    except Exception as e:
        return PlainTextResponse("", media_type="application/xml")


@app.get("/api/escalate/status/{ticket_id}")
def get_escalate_status(ticket_id: str):
    status = escalation_status.get(ticket_id, "WAITING")
    return {"ticket_id": ticket_id, "status": status}

import requests
from typing import List

# --- Chatbot Proxy Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: str

# --- Chatbot Endpoint ---
@app.post("/api/chat")
def chat_with_groq(request: ChatRequest):
    """Securely proxies the chat request to Groq, bypassing browser CORS."""
    
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build the exact message structure Groq expects
    groq_messages = [{"role": "system", "content": request.system_prompt}]
    for msg in request.messages:
        groq_messages.append({"role": msg.role, "content": msg.content})
        
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": groq_messages,
        "max_tokens": 300,
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status() # Check for errors
        
        reply_text = response.json()["choices"][0]["message"]["content"]
        return {"reply": reply_text}
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Groq API")
    
from fastapi import UploadFile, File, Form
import json
import requests

# --- Voice Chatbot Endpoint (Sarvam + Groq Pipeline) ---
@app.post("/api/voice-chat")
async def voice_chat_with_groq(
    file: UploadFile = File(...),
    messages: str = Form(...),
    system_prompt: str = Form(...)
):
    """Processes spoken audio, translates to English, gets Groq AI answer, and translates back."""
    
  
    
    # 1. AUDIO TO ENGLISH TEXT (via Sarvam STT)
    # ---------------------------------------------------------
    try:
        audio_bytes = await file.read()
        files = {"file": (file.filename, audio_bytes, file.content_type)}
        data = {
            "model": "saaras:v3",
            "mode": "translate"  # Directly translates regional speech to English text
        }
        headers = {"api-subscription-key": SARVAM_API_KEY}
        
        stt_res = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data)
        stt_res.raise_for_status()
        stt_data = stt_res.json()
        
        english_transcript = stt_data.get("transcript", "")
        detected_lang = stt_data.get("language_code", "en-IN")
        
        if not english_transcript:
            return {"reply": "I couldn't hear anything. Please try again.", "user_transcript": "No audio detected", "language": "unknown"}

    except Exception as e:
        print(f"Sarvam STT Error: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-Text conversion failed")

    # 2. GET EXPERT ANSWER (via Groq)
    # ---------------------------------------------------------
    try:
        # Parse the JSON string of messages sent from React
        history = json.loads(messages)
        
        groq_messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the new transcript as the latest user message
        groq_messages.append({"role": "user", "content": english_transcript})
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": groq_messages,
            "max_tokens": 300,
            "temperature": 0.2
        }
        
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        groq_res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=groq_headers, json=payload)
        groq_res.raise_for_status()
        english_ai_reply = groq_res.json()["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return {"reply": "I understood you, but my reasoning engine is down.", "user_transcript": english_transcript, "language": detected_lang}

    # 3. TRANSLATE BACK TO REGIONAL (via Sarvam Translate)
    # ---------------------------------------------------------
    final_reply = english_ai_reply
    
    # Only translate back if the user didn't speak English
    if detected_lang not in ["en-IN", "en-GB", "en-US", "unknown"]:
        try:
            trans_payload = {
                "input": english_ai_reply,
                "source_language_code": "en-IN",
                "target_language_code": detected_lang,
                "model": "sarvam-translate:v1"
            }
            
            trans_res = requests.post(
                "https://api.sarvam.ai/translate", 
                headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
                json=trans_payload
            )
            trans_data = trans_res.json()
            final_reply = trans_data.get("translated_text", english_ai_reply)
        except Exception as e:
            print(f"Translation Error: {e}")
            # Fallback to English reply if translation fails

    return {
        "user_transcript": english_transcript,
        "reply": final_reply,
        "language": detected_lang
    }