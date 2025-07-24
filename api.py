
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import torch
import joblib

app = FastAPI(title="ThermoSense ML + GPT-2 Advisory")

# ========== Load Model Assets ==========
print("Loading model and encoder...")

# Load CSV for encoder fitting (simulate original training data)
df = pd.read_csv("thermosense_test_data.csv")
features = ["battery_temp", "ambient_temp", "device_state"]
target = "measured_health_impact"

encoder = OneHotEncoder(sparse_output=False)
device_state_encoded = encoder.fit_transform(df[["device_state"]])
device_state_df = pd.DataFrame(device_state_encoded, columns=encoder.get_feature_names_out(["device_state"]))
X = pd.concat([df[["battery_temp", "ambient_temp"]].reset_index(drop=True), device_state_df], axis=1)
y = df[target]

# Fit the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# GPT-2 setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

FEW_SHOT_PROMPT = """You are a battery safety advisor. Here are some examples:

Report:
- Battery temperature: 41.3°C
- Ambient temperature: 38.0°C
- Device state: Charging
- Predicted battery health impact: 0.125
Advice: Danger: Device is overheating while charging. Unplug it and let it cool down immediately!

Report:
- Battery temperature: 34.9°C
- Ambient temperature: 28.5°C
- Device state: Discharging
- Predicted battery health impact: 0.019
Advice: All clear: Device temperature is normal. Continue using as usual.

Report:
- Battery temperature: 38.5°C
- Ambient temperature: 35.2°C
- Device state: Idle
- Predicted battery health impact: 0.087
Advice: Warning: Battery is getting warm. Move device to cooler location and avoid heavy usage.

Report:
- Battery temperature: {battery_temp:.1f}°C
- Ambient temperature: {ambient_temp:.1f}°C
- Device state: {device_state}
- Predicted battery health impact: {pred_impact:.3f}
Advice:"""

# ========== Input Schema ==========
class SensorInput(BaseModel):
    battery_temp: float
    ambient_temp: float
    device_state: str

# ========== Helper Functions ==========
def generate_advice_with_gpt2(battery_temp, ambient_temp, device_state, pred_impact):
    prompt = FEW_SHOT_PROMPT.format(
        battery_temp=battery_temp,
        ambient_temp=ambient_temp,
        device_state=device_state.capitalize(),
        pred_impact=pred_impact
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + 40,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    advice = decoded.split("Advice:")[-1].strip()
    if "." in advice:
        advice = advice.split(".")[0] + "."
    if not advice or len(advice) < 10:
        if pred_impact > 0.07:
            advice = "Critical: High battery stress detected. Take immediate action to cool device."
        elif pred_impact > 0.04:
            advice = "Warning: Moderate battery stress. Consider reducing usage."
        else:
            advice = "Normal: Battery conditions are within safe limits."
    return advice

def get_alert_level(impact):
    if impact > 0.07:
        return "danger"
    elif impact > 0.04:
        return "warning"
    else:
        return "safe"

# ========== API Endpoint ==========
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
        advice = generate_advice_with_gpt2(
            input.battery_temp, input.ambient_temp, input.device_state, impact
        )
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

# Uncomment if running directly
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
