"""
Workaround for FP8 quantization with sequential offloading on large models.
This patches the DataFreePipeline to skip the dispatch_for_generation call
that would load the entire model to GPU.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.helpers import dispatch_for_sequential
from llmcompressor.utils.helpers import patch_attr
from llmcompressor.pipelines.data_free import pipeline as datafree_module

MODEL_ID = "/home/arli/models/GLM-4.5-Air-Abliterated"

# Load model to CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Enable sequential offloading
dispatch_for_sequential(model)

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

# Patch DataFreePipeline to skip dispatch_for_generation
# This prevents it from trying to load the entire model to GPU
def noop_dispatch(model):
    """No-op dispatch function that doesn't change model placement"""
    return model

# Apply FP8 quantization with sequential offloading
with patch_attr(datafree_module, 'dispatch_for_generation', noop_dispatch):
    oneshot(model=model, recipe=recipe)

# Save quantized model
SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model saved to {SAVE_DIR}")