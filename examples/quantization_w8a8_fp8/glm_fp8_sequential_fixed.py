"""
FP8 quantization with sequential offloading for large MoE models.
This uses the new SequentialDataFreePipeline to enable sequential offloading
without requiring a calibration dataset.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/home/arli/models/GLM-4.5-Air-Abliterated"

# Load model to CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map=None  # Critical: loads to CPU for sequential offloading
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "Glm4MoeAttention",
    ],
)

# Apply FP8 quantization with sequential offloading
# The pipeline="datafree_sequential" parameter enables sequential offloading
# without requiring a calibration dataset
oneshot(
    model=model,
    recipe=recipe,
    pipeline="datafree_sequential",  # Use the new sequential data-free pipeline
)

# Save quantized model
SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model successfully quantized and saved to {SAVE_DIR}")