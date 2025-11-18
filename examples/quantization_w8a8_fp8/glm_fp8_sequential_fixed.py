"""
FP8 quantization with sequential offloading for GLM-4.5-Air MoE model.
This uses the new SequentialDataFreePipeline to enable sequential offloading
without requiring a calibration dataset.

Configuration matches the working vLLM-compatible quantization.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

MODEL_ID = "/home/arli/models/GLM-4.5-Air-Abliterated"

# Load model to CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map=None  # Critical: loads to CPU for sequential offloading
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Match the working quantization config exactly:
# - Input activations: FP8 dynamic per-token
# - Weights: FP8 static per-channel
# - Only ignore: router gates, biases, norms, embeddings, lm_head
recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            input_activations=QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=True,
                dynamic=True,
                strategy="token",
            ),
            weights=QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=True,
                dynamic=False,
                strategy="channel",
                observer="minmax",
            ),
        )
    },
    ignore=[
        # Only ignore router gates (not the expert gates/up_proj!)
        "re:.*mlp\\.gate$",  # Matches model.layers.X.mlp.gate (router)
        "re:.*mlp\\.gate\\..*",  # Matches model.layers.X.mlp.gate.e_score_correction_bias
        # Ignore biases and norms
        r"re:.*\.bias$",
        "re:.*norm$",
        # Ignore embeddings
        "re:.*embed.*",
        "lm_head",
        # GLM-specific layers from layer 46
        "re:.*eh_proj",
        "re:.*shared_head",
        "re:.*hnorm",
        "re:.*enorm",
    ],
)

# Apply FP8 quantization with sequential offloading
oneshot(
    model=model,
    recipe=recipe,
    pipeline="sequential_datafree",  # Use the new sequential data-free pipeline
)

# Save quantized model
SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model successfully quantized and saved to {SAVE_DIR}")