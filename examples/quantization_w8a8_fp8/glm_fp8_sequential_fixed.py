"""
FP8 quantization with sequential offloading for GLM-4.5-Air MoE model.
This uses the new SequentialDataFreePipeline to enable sequential offloading
without requiring a calibration dataset.

Uses the exact same ignore list as the original working model.
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

# Exact ignore list from the original working model
IGNORE_LIST = [
    "model.layers.9.mlp.gate", "model.layers.22.self_attn.q_proj.bias", "model.layers.21.post_attention_layernorm",
    "model.layers.46.post_attention_layernorm", "model.layers.33.mlp.gate.e_score_correction_bias", "model.layers.19.input_layernorm",
    "model.layers.23.input_layernorm", "model.layers.26.input_layernorm", "model.layers.16.input_layernorm", "model.layers.2.input_layernorm",
    "model.layers.43.mlp.gate.e_score_correction_bias", "model.layers.17.self_attn.v_proj.bias", "model.layers.4.self_attn.k_proj.bias",
    "model.layers.24.mlp.gate", "model.layers.42.self_attn.q_proj.bias", "model.layers.19.self_attn.v_proj.bias", "model.layers.36.self_attn.k_proj.bias",
    "model.layers.43.self_attn.k_proj.bias", "model.layers.26.self_attn.v_proj.bias", "model.layers.33.mlp.gate", "model.layers.31.input_layernorm",
    "model.layers.8.input_layernorm", "model.layers.9.mlp.gate.e_score_correction_bias", "model.layers.19.mlp.gate.e_score_correction_bias",
    "model.layers.14.self_attn.v_proj.bias", "model.layers.19.mlp.gate", "model.layers.12.mlp.gate", "model.layers.37.mlp.gate.e_score_correction_bias",
    "model.layers.39.mlp.gate", "model.layers.20.self_attn.v_proj.bias", "model.layers.1.self_attn.q_proj.bias", "model.layers.7.self_attn.k_proj.bias",
    "model.layers.19.self_attn.k_proj.bias", "model.layers.30.mlp.gate.e_score_correction_bias", "model.layers.37.input_layernorm",
    "model.layers.40.self_attn.q_proj.bias", "model.layers.30.input_layernorm", "model.layers.10.post_attention_layernorm", "model.layers.10.mlp.gate",
    "model.layers.21.mlp.gate", "model.layers.39.post_attention_layernorm", "model.layers.20.self_attn.q_proj.bias", "model.layers.20.input_layernorm",
    "model.layers.16.self_attn.v_proj.bias", "model.layers.5.self_attn.k_proj.bias", "model.layers.18.input_layernorm", "model.layers.35.input_layernorm",
    "model.layers.43.mlp.gate", "model.layers.5.self_attn.v_proj.bias", "model.layers.28.mlp.gate", "model.layers.4.self_attn.v_proj.bias",
    "model.layers.46.mlp.gate.e_score_correction_bias", "model.layers.25.mlp.gate.e_score_correction_bias", "model.layers.46.enorm",
    "model.layers.30.self_attn.k_proj.bias", "model.layers.27.self_attn.v_proj.bias", "model.layers.23.mlp.gate", "model.layers.38.post_attention_layernorm",
    "model.layers.0.post_attention_layernorm", "model.layers.46.self_attn.k_proj.bias", "model.layers.26.post_attention_layernorm",
    "model.layers.26.self_attn.q_proj.bias", "model.layers.46.embed_tokens", "model.layers.24.input_layernorm", "model.layers.41.post_attention_layernorm",
    "model.layers.20.mlp.gate", "model.layers.3.mlp.gate", "model.layers.22.input_layernorm", "model.layers.15.post_attention_layernorm",
    "model.layers.39.input_layernorm", "model.layers.42.mlp.gate.e_score_correction_bias", "model.layers.34.mlp.gate", "model.layers.13.mlp.gate",
    "model.layers.38.input_layernorm", "model.layers.15.self_attn.q_proj.bias", "model.layers.7.post_attention_layernorm", "model.layers.28.self_attn.v_proj.bias",
    "model.layers.36.post_attention_layernorm", "model.layers.34.self_attn.k_proj.bias", "model.layers.23.post_attention_layernorm",
    "model.layers.43.input_layernorm", "model.layers.39.self_attn.k_proj.bias", "model.layers.18.post_attention_layernorm",
    "model.layers.29.mlp.gate.e_score_correction_bias", "model.layers.34.mlp.gate.e_score_correction_bias", "model.layers.28.self_attn.q_proj.bias",
    "model.layers.46.eh_proj", "model.layers.14.mlp.gate.e_score_correction_bias", "model.layers.7.mlp.gate.e_score_correction_bias",
    "model.layers.35.self_attn.k_proj.bias", "model.layers.13.post_attention_layernorm", "model.layers.22.self_attn.k_proj.bias",
    "model.layers.4.self_attn.q_proj.bias", "model.layers.41.self_attn.k_proj.bias", "model.layers.12.post_attention_layernorm",
    "model.layers.37.self_attn.q_proj.bias", "model.layers.46.input_layernorm", "model.layers.24.self_attn.k_proj.bias", "model.layers.5.mlp.gate",
    "model.layers.9.self_attn.k_proj.bias", "model.layers.10.self_attn.v_proj.bias", "model.layers.42.self_attn.v_proj.bias", "model.embed_tokens",
    "model.layers.2.self_attn.q_proj.bias", "model.layers.28.mlp.gate.e_score_correction_bias", "model.layers.24.self_attn.v_proj.bias",
    "model.layers.15.input_layernorm", "model.layers.9.input_layernorm", "model.layers.33.input_layernorm", "model.layers.45.self_attn.v_proj.bias",
    "model.layers.31.self_attn.q_proj.bias", "model.layers.34.input_layernorm", "model.layers.14.input_layernorm", "model.layers.17.post_attention_layernorm",
    "model.layers.0.self_attn.k_proj.bias", "model.layers.37.self_attn.v_proj.bias", "model.norm", "model.layers.9.self_attn.q_proj.bias",
    "model.layers.4.input_layernorm", "model.layers.45.self_attn.q_proj.bias", "model.layers.7.self_attn.q_proj.bias", "model.layers.32.self_attn.v_proj.bias",
    "model.layers.22.self_attn.v_proj.bias", "model.layers.45.post_attention_layernorm", "model.layers.40.mlp.gate", "model.layers.29.self_attn.v_proj.bias",
    "model.layers.3.mlp.gate.e_score_correction_bias", "model.layers.31.post_attention_layernorm", "model.layers.41.self_attn.v_proj.bias",
    "model.layers.5.input_layernorm", "model.layers.13.self_attn.v_proj.bias", "model.layers.26.self_attn.k_proj.bias", "model.layers.28.post_attention_layernorm",
    "model.layers.17.mlp.gate", "model.layers.42.mlp.gate", "model.layers.34.self_attn.v_proj.bias", "model.layers.1.mlp.gate.e_score_correction_bias",
    "model.layers.21.input_layernorm", "model.layers.21.self_attn.k_proj.bias", "model.layers.29.self_attn.k_proj.bias", "model.layers.20.post_attention_layernorm",
    "model.layers.14.post_attention_layernorm", "model.layers.34.post_attention_layernorm", "model.layers.27.self_attn.k_proj.bias",
    "model.layers.24.mlp.gate.e_score_correction_bias", "model.layers.31.mlp.gate.e_score_correction_bias", "model.layers.2.self_attn.k_proj.bias",
    "model.layers.25.self_attn.v_proj.bias", "model.layers.1.post_attention_layernorm", "model.layers.10.self_attn.q_proj.bias",
    "model.layers.16.mlp.gate.e_score_correction_bias", "model.layers.16.self_attn.q_proj.bias", "model.layers.38.mlp.gate.e_score_correction_bias",
    "model.layers.46.self_attn.q_proj.bias", "model.layers.23.self_attn.k_proj.bias", "model.layers.42.post_attention_layernorm",
    "model.layers.33.self_attn.k_proj.bias", "model.layers.30.mlp.gate", "model.layers.34.self_attn.q_proj.bias", "model.layers.4.post_attention_layernorm",
    "model.layers.13.self_attn.k_proj.bias", "model.layers.2.post_attention_layernorm", "model.layers.40.post_attention_layernorm",
    "model.layers.38.self_attn.k_proj.bias", "model.layers.1.self_attn.k_proj.bias", "model.layers.10.mlp.gate.e_score_correction_bias",
    "model.layers.43.self_attn.v_proj.bias", "model.layers.11.input_layernorm", "model.layers.42.input_layernorm", "model.layers.19.self_attn.q_proj.bias",
    "model.layers.24.post_attention_layernorm", "model.layers.12.input_layernorm", "model.layers.42.self_attn.k_proj.bias", "model.layers.12.self_attn.k_proj.bias",
    "model.layers.0.self_attn.v_proj.bias", "model.layers.1.mlp.gate", "model.layers.39.self_attn.v_proj.bias", "model.layers.14.mlp.gate",
    "model.layers.44.post_attention_layernorm", "model.layers.37.mlp.gate", "model.layers.31.mlp.gate", "model.layers.8.post_attention_layernorm",
    "model.layers.2.mlp.gate.e_score_correction_bias", "model.layers.36.input_layernorm", "model.layers.30.post_attention_layernorm",
    "model.layers.46.shared_head.norm", "model.layers.4.mlp.gate", "model.layers.6.mlp.gate", "model.layers.29.mlp.gate", "model.layers.7.mlp.gate",
    "model.layers.0.self_attn.q_proj.bias", "model.layers.44.mlp.gate", "model.layers.32.self_attn.k_proj.bias", "model.layers.4.mlp.gate.e_score_correction_bias",
    "model.layers.18.self_attn.v_proj.bias", "model.layers.30.self_attn.q_proj.bias", "model.layers.21.mlp.gate.e_score_correction_bias",
    "model.layers.32.post_attention_layernorm", "model.layers.19.post_attention_layernorm", "model.layers.22.mlp.gate", "model.layers.13.mlp.gate.e_score_correction_bias",
    "model.layers.8.mlp.gate", "model.layers.36.self_attn.v_proj.bias", "model.layers.5.post_attention_layernorm", "model.layers.32.input_layernorm",
    "model.layers.33.post_attention_layernorm", "model.layers.21.self_attn.v_proj.bias", "model.layers.2.mlp.gate", "model.layers.13.input_layernorm",
    "model.layers.15.self_attn.v_proj.bias", "model.layers.16.self_attn.k_proj.bias", "model.layers.2.self_attn.v_proj.bias", "model.layers.43.post_attention_layernorm",
    "model.layers.7.input_layernorm", "model.layers.29.post_attention_layernorm", "model.layers.20.self_attn.k_proj.bias", "model.layers.38.mlp.gate",
    "model.layers.18.mlp.gate.e_score_correction_bias", "model.layers.25.input_layernorm", "model.layers.1.input_layernorm", "model.layers.46.hnorm",
    "model.layers.31.self_attn.v_proj.bias", "model.layers.14.self_attn.q_proj.bias", "model.layers.18.self_attn.q_proj.bias", "model.layers.8.self_attn.q_proj.bias",
    "model.layers.35.self_attn.v_proj.bias", "model.layers.45.mlp.gate.e_score_correction_bias", "model.layers.9.post_attention_layernorm",
    "model.layers.30.self_attn.v_proj.bias", "model.layers.15.mlp.gate", "model.layers.10.input_layernorm", "model.layers.6.self_attn.q_proj.bias",
    "model.layers.11.mlp.gate.e_score_correction_bias", "model.layers.41.input_layernorm", "model.layers.22.mlp.gate.e_score_correction_bias",
    "model.layers.15.mlp.gate.e_score_correction_bias", "model.layers.21.self_attn.q_proj.bias", "model.layers.17.mlp.gate.e_score_correction_bias",
    "model.layers.16.mlp.gate", "model.layers.25.self_attn.q_proj.bias", "model.layers.6.input_layernorm", "model.layers.17.input_layernorm",
    "model.layers.26.mlp.gate.e_score_correction_bias", "model.layers.35.mlp.gate.e_score_correction_bias", "model.layers.0.input_layernorm",
    "model.layers.3.post_attention_layernorm", "model.layers.6.self_attn.v_proj.bias", "model.layers.27.mlp.gate.e_score_correction_bias",
    "model.layers.18.mlp.gate", "model.layers.28.input_layernorm", "model.layers.9.self_attn.v_proj.bias", "model.layers.31.self_attn.k_proj.bias",
    "model.layers.40.self_attn.v_proj.bias", "model.layers.12.self_attn.q_proj.bias", "model.layers.41.mlp.gate", "model.layers.5.self_attn.q_proj.bias",
    "model.layers.11.self_attn.v_proj.bias", "model.layers.36.mlp.gate", "model.layers.27.self_attn.q_proj.bias", "model.layers.40.self_attn.k_proj.bias",
    "model.layers.11.post_attention_layernorm", "model.layers.27.input_layernorm", "model.layers.12.self_attn.v_proj.bias", "model.layers.46.mlp.gate",
    "model.layers.17.self_attn.k_proj.bias", "model.layers.3.input_layernorm", "model.layers.44.input_layernorm", "model.layers.10.self_attn.k_proj.bias",
    "model.layers.41.mlp.gate.e_score_correction_bias", "model.layers.7.self_attn.v_proj.bias", "model.layers.18.self_attn.k_proj.bias",
    "model.layers.1.self_attn.v_proj.bias", "model.layers.26.mlp.gate", "model.layers.45.input_layernorm", "model.layers.23.self_attn.v_proj.bias",
    "model.layers.39.mlp.gate.e_score_correction_bias", "model.layers.12.mlp.gate.e_score_correction_bias", "model.layers.37.post_attention_layernorm",
    "model.layers.46.self_attn.v_proj.bias", "model.layers.36.mlp.gate.e_score_correction_bias", "model.layers.5.mlp.gate.e_score_correction_bias",
    "model.layers.35.mlp.gate", "model.layers.44.self_attn.k_proj.bias", "model.layers.3.self_attn.k_proj.bias", "model.layers.11.mlp.gate",
    "model.layers.11.self_attn.q_proj.bias", "model.layers.17.self_attn.q_proj.bias", "model.layers.32.self_attn.q_proj.bias",
    "model.layers.11.self_attn.k_proj.bias", "model.layers.40.mlp.gate.e_score_correction_bias", "model.layers.41.self_attn.q_proj.bias",
    "model.layers.15.self_attn.k_proj.bias", "model.layers.44.self_attn.v_proj.bias", "model.layers.25.self_attn.k_proj.bias",
    "model.layers.25.post_attention_layernorm", "model.layers.29.input_layernorm", "model.layers.44.self_attn.q_proj.bias", "model.layers.16.post_attention_layernorm",
    "model.layers.6.mlp.gate.e_score_correction_bias", "model.layers.38.self_attn.v_proj.bias", "model.layers.40.input_layernorm",
    "model.layers.6.post_attention_layernorm", "model.layers.22.post_attention_layernorm", "model.layers.8.self_attn.k_proj.bias",
    "model.layers.37.self_attn.k_proj.bias", "model.layers.23.mlp.gate.e_score_correction_bias", "model.layers.27.mlp.gate", "model.layers.8.mlp.gate.e_score_correction_bias",
    "model.layers.28.self_attn.k_proj.bias", "model.layers.24.self_attn.q_proj.bias", "model.layers.39.self_attn.q_proj.bias", "model.layers.36.self_attn.q_proj.bias",
    "model.layers.45.self_attn.k_proj.bias", "model.layers.32.mlp.gate.e_score_correction_bias", "model.layers.35.self_attn.q_proj.bias",
    "model.layers.33.self_attn.q_proj.bias", "model.layers.14.self_attn.k_proj.bias", "lm_head", "model.layers.3.self_attn.v_proj.bias",
    "model.layers.44.mlp.gate.e_score_correction_bias", "model.layers.45.mlp.gate", "model.layers.32.mlp.gate", "model.layers.33.self_attn.v_proj.bias",
    "model.layers.29.self_attn.q_proj.bias", "model.layers.3.self_attn.q_proj.bias", "model.layers.35.post_attention_layernorm", "model.layers.6.self_attn.k_proj.bias",
    "model.layers.43.self_attn.q_proj.bias", "model.layers.20.mlp.gate.e_score_correction_bias", "model.layers.8.self_attn.v_proj.bias",
    "model.layers.13.self_attn.q_proj.bias", "model.layers.27.post_attention_layernorm", "model.layers.38.self_attn.q_proj.bias", "model.layers.25.mlp.gate",
    "model.layers.23.self_attn.q_proj.bias"
]

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
    ignore=IGNORE_LIST,
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