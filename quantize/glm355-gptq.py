import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

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

# 4. Configure Mixed Precision Int4/Int8 Recipe
# This applies int8 to previously ignored layers (except lm_head), int4 to everything else
recipe = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: int
                        strategy: group
                        dynamic: false
                        symmetric: true
                        group_size: 128
                    targets: ["model.embed_tokens", "re:.*shared_experts.*proj.*", "re:.*self_attn.(q|k|v|o)_proj.*", "re:.*layers.[0-3].*proj.*", "re:.*layers.89.*proj.*", "re:.*layers.9[0-2].*proj.*"]
                group_1:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: group
                        dynamic: false
                        symmetric: false
                        group_size: 128
                    targets: ["Linear"]
"""

# 5. Apply Mixed Precision Quantization
# sequential_targets=["GLMBlock"] ensures we only load one layer block onto
# the GPU at a time to save VRAM.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES
)

# 6. Save Compressed Model
SAVE_DIR = MODEL_ID + "-GPTQ-INT4-INT8-Mixed"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model saved to {SAVE_DIR}")