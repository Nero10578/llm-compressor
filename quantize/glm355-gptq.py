import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 1. Configuration
MODEL_ID = "/home/arli/models/GLM-4.6-Derestricted" # Or your specific GLM ID
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 4096

# 2. Load Model & Tokenizer
# We load in standard precision (BF16 or FP16) to calibrate
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 3. Prepare Calibration Dataset
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# 4. Define Ignore Rules (Based on your JSON config)
# We use "re:" prefix to tell llmcompressor to treat strings as regex
ignores = [
    # --- Standard Excludes (-:) ---
    "lm_head",
    "model.embed_tokens",
    r"re:.*shared_experts.*",  # Regex for all shared experts
    r"re:.*shared_head.*",     # Regex for shared head
    r"re:.*self_attn.*",
    
    # --- Mixed Precision Layers (+:) ---
    # The config sets layers 0-4 and 44-46 to FP16.
    # We ignore them here to keep them at FP16 (safest for W4A16 recipe).
    # [0-4] matches 0,1,2,3,4
    # 8[8-9]|9[0-2] matches 88,89,90,91,92
    r"re:model\.layers\.(?:[0-3]|89|9[0-2])\..*",
]

# 5. Configure GPTQ Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",        # 4-bit Weights, 16-bit Activations
    ignore=ignores,
)

# 6. Apply Quantization
# sequential_targets=["GLMBlock"] ensures we only load one layer block onto
# the GPU at a time to save VRAM.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES
)

# 7. Save Compressed Model
SAVE_DIR = MODEL_ID + "-GPTQ-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model saved to {SAVE_DIR}")