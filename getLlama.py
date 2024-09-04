import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Load the model and tokenizer from Hugging Face
model_name = "amaricem/Meta-Llama-3.1-8B-pgt-v1"

# Adjust these settings based on your hardware and preferences
max_seq_length = 512
dtype = torch.float16  # Use torch.float16 for T4/V100, bfloat16 for Ampere+ if supported

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with 4-bit quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map the model to available devices
    torch_dtype=dtype,  # Set the dtype for the model
    load_in_4bit=False  # Load the model with 4-bit quantization
)

# Define the prompt
polygt_prompt = """Unten finden Sie eine Frage. Reagieren Sie mit nur einer zutreffenden Antwort.

###Instruction:
{instruction}

###Input:
{input}

###Response:
{response}"""

# Prepare the input text
input_text = polygt_prompt.format(
    instruction="Wann gehst du ins Kaufhaus?",  # instruction
    input="",  # input
    response=""  # output - leave this blank for generation!
)

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", max_length=max_seq_length, truncation=True)

# Move inputs to the appropriate device
inputs = {key: value.to("cpu") for key, value in inputs.items()}

# Generate the output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )

# Decode and print the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
