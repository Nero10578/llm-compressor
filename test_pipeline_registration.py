"""Test that the new pipeline is registered correctly"""
from llmcompressor.pipelines.registry import CalibrationPipeline

# This will show all registered pipelines
print("Registered pipelines:")
print(CalibrationPipeline._registry)

# Try to load the new pipeline
try:
    pipeline = CalibrationPipeline.load_from_registry("datafree_sequential")
    print("\n✓ Successfully loaded datafree_sequential pipeline!")
    print(f"Pipeline class: {pipeline}")
except KeyError as e:
    print(f"\n✗ Failed to load pipeline: {e}")