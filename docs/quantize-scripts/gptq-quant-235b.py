import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import random

# Config
# IMPORTANT: Load the smoothed model from the intermediate directory
MODEL_ID = "/home/arli/models/Q3-235B-RpR-V2-8e-6-smoothed"
SAVE_DIR = "/home/arli/models/Q3-235B-RpR-V2-8e-6-W4A16-gptq"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 128

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and preprocess the dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=1337).select(range(NUM_CALIBRATION_SAMPLES))

def add_system_prompt(messages):
    options = ["", "You are a helpful assistant."]
    thinking = random.choice(options)
    return [{"content": f"{thinking}", "role": "system"}] + messages

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(add_system_prompt(example["messages"]), tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithms
recipe = [
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"], dampening_frac=0.01, offload_hessians=True),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True
)

# Save the compressed model
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
