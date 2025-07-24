from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load T5 model & tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def generate_advice_with_t5(battery_temp, ambient_temp, device_state, pred_impact):
    prompt = (
        f"Given the following parameters:\n"
        f"- Battery temperature: {battery_temp:.1f}°C\n"
        f"- Ambient temperature: {ambient_temp:.1f}°C\n"
        f"- Device state: {device_state}\n"
        f"- Predicted battery health impact: {pred_impact:.3f}\n"
        "\nGenerate a one-sentence actionable battery safety tip for the end user."
    )
    # Tokenize and generate
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=40)
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return advice

# Example integration (assuming regressor predicts 0.085)
advice = generate_advice_with_t5(38.9, 32.2, 'charging', 0.095)
print("T5 Advice:", advice)