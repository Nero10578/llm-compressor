"""
Helper script to inspect GLM model structure and identify Linear layers
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "/home/arli/models/GLM-4.6-Derestricted"

print("Loading model to inspect layer structure...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="cpu")

print("\n=== MODEL STRUCTURE INSPECTION ===")
print("Looking for Linear layers and other layer types...")

linear_layers = []
layernorm_layers = []
other_layers = []

def inspect_module(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, torch.nn.Linear):
            linear_layers.append(full_name)
            print(f"LINEAR: {full_name}")
        elif "layernorm" in name.lower() or "norm" in name.lower():
            layernorm_layers.append(full_name)
            print(f"LAYERNORM: {full_name}")
        elif hasattr(child, 'weight') and hasattr(child, 'bias'):
            if len(list(child.children())) == 0:  # Leaf module with weights
                other_layers.append(f"{full_name} ({type(child).__name__})")
                print(f"OTHER: {full_name} ({type(child).__name__})")
        
        if len(list(child.children())) > 0:
            inspect_module(child, full_name)

inspect_module(model)

# Print layers information into a text file
with open("glm_layers.txt", "w") as f:
    f.write(f"Total Linear layers: {len(linear_layers)}\n")
    f.write(f"Total LayerNorm layers: {len(layernorm_layers)}\n")
    f.write(f"Other layers with weights: {len(other_layers)}\n")
    f.write("\n=== LINEAR LAYERS ===\n")
    for layer in linear_layers:
        f.write(f"  {layer}\n")
    f.write("\n=== LAYERNORM LAYERS ===\n")
    for layer in layernorm_layers:
        f.write(f"  {layer}\n")
    f.write("\n=== OTHER WEIGHTED LAYERS ===\n")
    for layer in other_layers:
        f.write(f"  {layer}\n")