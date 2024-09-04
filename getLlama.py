import torch

try:
    from unsloth import FastLanguageModel
    print("Unsloth loaded successfully")
except AssertionError as e:
    if "CUDA" in str(e):
        print("CUDA is not available, switching to CPU-only mode.")
    else:
        raise e

# Set your configurations
max_seq_length = 512  # You can adjust this based on your memory
dtype = torch.float32  # Set dtype based on your GPU; use 'float16' for T4, V100, or 'bfloat16' for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage, or set to False

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="amaricem/Meta-Llama-3.1-8B-pgt-v1",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="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",  # Token required for gated models
)

# Define the prompt
polygt_prompt = """Unten finden Sie eine Frage. Reagieren Sie mit nur einer zutreffenden Antwort.

###Instruction:
{instruction}

###Input:
{input}

###Response:
{response}"""

# Get the EOS token from the tokenizer
EOS_TOKEN = tokenizer.eos_token

# Enable faster inference with FastLanguageModel (no CUDA dependency)
FastLanguageModel.for_inference(model)

# Prepare inputs for the model
inputs = tokenizer(
    [
        polygt_prompt.format(
            instruction="Wann gehst du ins kaufhaus?",  # instruction
            input="",  # input
            response=""  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt"
).to("cpu")  

# Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    use_cache=True,
)

# Decode and print the output
decoded_output = tokenizer.batch_decode(outputs)
print(decoded_output)
