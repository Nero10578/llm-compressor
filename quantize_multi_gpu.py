import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import random

# Config
MODEL_ID = "/home/arli/models/Llama-3_1-Nemotron-Ultra-253B-v1"
SAVE_DIR = "/home/arli/models/Llama-3_1-Nemotron-Ultra-253B-v1-W4A16"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 64

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    offload_folder="./offload",
    max_memory={i: "22GiB" for i in range(torch.cuda.device_count())},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and preprocess the dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=1337).select(range(NUM_CALIBRATION_SAMPLES))

def add_system_prompt(messages):
    options = ["on", "off"]
    thinking = random.choice(options)
    return [{"content": f"detailed thinking {thinking}", "role": "system"}] + messages

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(add_system_prompt(example["messages"]), tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithms
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*125.*", "re:.*134.*", "re:.*143.*", "re:.*149.*"], dampening_frac=0.01, offload_hessians=True),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    pipeline="basic"
)

# Save the compressed model
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
