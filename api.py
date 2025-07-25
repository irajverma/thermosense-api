import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import requests

app = FastAPI(title="ThermoSense ML + GPT-2 Advisory")

# ========== Load Model and Encoder ==========
print("🔧 Loading model and encoder...")

df = pd.read_csv("thermosense_test_data.csv")
features = ["battery_temp", "ambient_temp", "device_state"]
target = "measured_health_impact"

encoder = OneHotEncoder(sparse_output=False)
device_state_encoded = encoder.fit_transform(df[["device_state"]])
device_state_df = pd.DataFrame(device_state_encoded, columns=encoder.get_feature_names_out(["device_state"]))
X = pd.concat([df[["battery_temp", "ambient_temp"]].reset_index(drop=True), device_state_df], axis=1)
y = df[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ========== FastAPI Input Schema ==========
class SensorInput(BaseModel):
    battery_temp: float
    ambient_temp: float
    device_state: str

# ========== Helper Functions ==========
def generate_advice_with_gpt2(battery_temp, ambient_temp, device_state, pred_impact):
    try:
        response = requests.post(
            "https://iRajVerma-thermosense-gradio.hf.space/run/predict",  # your Hugging Face Space endpoint
            json={
                "battery_temp": battery_temp,
                "ambient_temp": ambient_temp,
                "device_state": device_state,
                "pred_impact": pred_impact
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("natural_language_tip", "Advice unavailable.")
        else:
            return "⚠️ GPT-2 service unavailable. Using fallback logic."
    except Exception as e:
        print("Error contacting GPT-2 API:", e)
        return "⚠️ GPT-2 service failed. Using fallback."

def get_alert_level(impact):
    if impact > 0.07:
        return "danger"
    elif impact > 0.04:
        return "warning"
    else:
        return "safe"

# ========== API Endpoints ==========
@app.get("/")
def home():
    return {"message": "Welcome to ThermoSense Advisory API. Use POST /api/advice to get predictions."}

@app.post("/api/advice")
def get_advice(input: SensorInput):
    try:
        input_df = pd.DataFrame([input.dict()])
        encoded_state = encoder.transform(input_df[["device_state"]])
        encoded_df = pd.DataFrame(encoded_state, columns=encoder.get_feature_names_out(["device_state"]))
        X_live = pd.concat([input_df[["battery_temp", "ambient_temp"]].reset_index(drop=True), encoded_df], axis=1)
        X_live = X_live.reindex(columns=X.columns, fill_value=0)

        impact = model.predict(X_live)[0]
        alert = get_alert_level(impact)
        advice = generate_advice_with_gpt2(input.battery_temp, input.ambient_temp, input.device_state, impact)

        action = None
        if alert == "danger":
            action = "Stop using device immediately and allow cooling"
        elif alert == "warning":
            action = "Monitor temperature and reduce intensive tasks"

        return {
            "predicted_health_impact": round(float(impact), 5),
            "alert_level": alert,
            "natural_language_tip": advice,
            "optional_action": action
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
