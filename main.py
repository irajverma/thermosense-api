
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 1. Load the CSV file
df = pd.read_csv('/Users/divyarthjain/Documents/ML_Projects/ML_Project_folder/thermosense_test_data.csv')

# 2. Preprocess: Features and target selection
features = ['battery_temp', 'ambient_temp', 'device_state']
target = 'measured_health_impact'

# One-hot encode categorical variable
encoder = OneHotEncoder(sparse_output=False)
device_state_encoded = encoder.fit_transform(df[['device_state']])
device_state_df = pd.DataFrame(device_state_encoded, columns=encoder.get_feature_names_out(['device_state']))
X = pd.concat([df[['battery_temp', 'ambient_temp']].reset_index(drop=True), device_state_df], axis=1)
y = df[target]

# 3. Split and Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Load GPT-2 model and tokenizer
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpt2.eval()

# 5. Few-shot prompt template for better GPT-2 performance
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

# 6. GPT-2 advice generation function with fixed attention mask
def generate_advice_with_gpt2(battery_temp, ambient_temp, device_state, pred_impact):
    """Generate battery safety advice using GPT-2 with proper attention mask."""
    
    # Format the prompt
    prompt = FEW_SHOT_PROMPT.format(
        battery_temp=battery_temp,
        ambient_temp=ambient_temp,
        device_state=device_state.capitalize(),
        pred_impact=pred_impact
    )
    
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate with proper attention mask
    with torch.no_grad():
        generated_ids = model_gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # This fixes the warning
            max_length=input_ids.shape[1] + 40,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode the generated text
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract only the advice part (after the last "Advice:")
    advice = output.split('Advice:')[-1].strip()
    
    # Clean up: take only the first sentence
    if '.' in advice:
        advice = advice.split('.')[0].strip() + '.'
    
    # Ensure advice is not empty
    if not advice or len(advice) < 10:
        # Fallback advice based on impact level
        if pred_impact > 0.07:
            advice = "Critical: High battery stress detected. Take immediate action to cool device."
        elif pred_impact > 0.04:
            advice = "Warning: Moderate battery stress. Consider reducing usage."
        else:
            advice = "Normal: Battery conditions are within safe limits."
    
    return advice

# 7. Determine alert level based on health impact
def get_alert_level(health_impact):
    """Determine alert level based on predicted health impact."""
    if health_impact > 0.07:
        return "danger"
    elif health_impact > 0.04:
        return "warning"
    else:
        return "safe"

# 8. Main advisory service function
def advisory_service(input_row):
    """Generate battery health advisory for given input."""
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_row])
    
    # Encode device state
    device_state_enc = encoder.transform(input_df[['device_state']])
    device_state_enc_df = pd.DataFrame(
        device_state_enc, 
        columns=encoder.get_feature_names_out(['device_state'])
    )
    
    # Combine features
    X_live = pd.concat([
        input_df[['battery_temp', 'ambient_temp']].reset_index(drop=True), 
        device_state_enc_df
    ], axis=1)
    
    # Ensure column order matches training data
    X_live = X_live.reindex(columns=X_train.columns, fill_value=0)
    
    # Predict health impact
    health_impact = model.predict(X_live)[0]
    
    # Generate advice using GPT-2
    advice = generate_advice_with_gpt2(
        input_row['battery_temp'],
        input_row['ambient_temp'],
        input_row['device_state'],
        health_impact
    )
    
    # Determine alert level
    alert_level = get_alert_level(health_impact)
    
    # Determine optional action
    if alert_level == "danger":
        action = "Stop using device immediately and allow cooling"
    elif alert_level == "warning":
        action = "Monitor temperature and reduce intensive tasks"
    else:
        action = None
    
    return {
        "alert_level": alert_level,
        "natural_language_tip": advice,
        "optional_action": action,
        "predicted_health_impact": round(float(health_impact), 5)
    }

# 9. Demo: Test the advisory service
if __name__ == "__main__":
    print("ThermoSense Advisory Service - GPT-2 Based\n")
    print("=" * 60)
    
    # Test on 3 random samples
    test_samples = df.sample(3, random_state=42)
    
    for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
        input_row = {
            'battery_temp': row['battery_temp'],
            'ambient_temp': row['ambient_temp'],
            'device_state': row['device_state']
        }
        
        print(f"\nSample {idx}:")
        print(f"Input: Battery={input_row['battery_temp']:.1f}°C, "
              f"Ambient={input_row['ambient_temp']:.1f}°C, "
              f"State={input_row['device_state']}")
        
        advisory = advisory_service(input_row)
        print(f"Output:")
        print(json.dumps(advisory, indent=2))
        print("-" * 40)