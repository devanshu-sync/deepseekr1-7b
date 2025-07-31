import runpod
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"
os.makedirs("/runpod-volume/hf_cache", exist_ok=True)

# Model and tokenizer loading
model_name = "google/gemma-3-12b-it"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load model in 8-bit for VRAM savings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,  # Quantization
    low_cpu_mem_usage=True
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

def run_model(prompt):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def handler(event):
    print("Worker Start")
    user_input = event.get("input", {})
    prompt = user_input.get("prompt")

    if not prompt:
        return {"status": "error", "message": "Missing 'prompt' in input"}

    try:
        result = run_model(prompt)
        return {
            "status": "success",
            "prompt": prompt,
            "generated_text": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})
