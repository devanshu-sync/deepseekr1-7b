from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the Gemma 3 4B instruction-tuned model
model_name = "google/gemma-3-4b"

local_path = "./gemma3_4b_model"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(local_path)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(local_path)


