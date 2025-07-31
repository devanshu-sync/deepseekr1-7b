from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

model_name = "google/gemma-3-12b-it"
local_path = "./gemma-3-12b-it-model"

# BitsAndBytes config for 8-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Download tokenizer and save
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(local_path)

# Download model in 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config
)
model.save_pretrained(local_path)
