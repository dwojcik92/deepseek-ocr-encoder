"""
Basic usage example for DeepSeek OCR Encoder.

This script demonstrates:
1. Simple one-line initialization with from_pretrained() (Recommended)
2. Advanced manual model loading (for more control)
3. Encoding images to vision tokens
"""

import os
import torch
from transformers import AutoModel
from PIL import Image
from deepseek_ocr_encoder import DeepSeekOCREncoder


def simple_usage_example():
    """Demonstrate the simple one-line initialization (Recommended)."""
    print("=" * 60)
    print("SIMPLE USAGE (Recommended)")
    print("=" * 60)
    
    # One-line initialization - automatically handles device, dtype, and model loading
    print("Loading encoder with from_pretrained()...")
    encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")
    print(f"✓ Encoder loaded! Using device: {encoder.device}, dtype: {encoder.dtype}")
    
    # Optional: Capture CUDA graph for faster inference (CUDA only)
    if encoder.device.type == "cuda":
        print("\nCapturing CUDA graph for optimized inference...")
        encoder.capture_cudagraph(batch_size=1, H=1024, W=1024)
        print("✓ CUDA graph captured!")
    
    # Example: Encode an image
    image_path = "example_image.png"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"\nEncoding image: {image_path}")
        vision_tokens = encoder(image_path)
        print(f"✓ Vision tokens shape: {vision_tokens.shape}")
        print(f"✓ Vision tokens dtype: {vision_tokens.dtype}")
    else:
        print(f"\nWarning: Example image not found at {image_path}")
        print("Please provide a valid image path to test encoding.")
    
    return encoder


def advanced_usage_example():
    """Demonstrate advanced manual model loading for more control."""
    print("\n" + "=" * 60)
    print("ADVANCED USAGE (Manual Model Loading)")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the base DeepSeek-OCR model manually
    print("\nLoading DeepSeek-OCR model manually...")
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    )
    model = model.eval().to(device)
    print("✓ Model loaded successfully!")

    # Create the optimized encoder with manual configuration
    print("\nCreating optimized encoder with custom settings...")
    encoder = DeepSeekOCREncoder(
        full_model=model,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        freeze=True,
        eager_to_device=True,
        precompute_pos_for_1024=True,
        use_compile=False,  # Set True for PyTorch 2.3+ with extra fusion
    )
    print("✓ Encoder created!")

    # Optional: Capture CUDA graph for faster inference (CUDA only)
    if device == "cuda":
        print("\nCapturing CUDA graph for optimized inference...")
        encoder.capture_cudagraph(batch_size=1, H=1024, W=1024)
        print("✓ CUDA graph captured!")

    # Example: Encode an image
    image_path = "example_image.png"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"\nEncoding image: {image_path}")
        
        # Method 1: Encode from path
        vision_tokens = encoder.encode(image_path)
        print(f"✓ Vision tokens shape: {vision_tokens.shape}")
        print(f"✓ Vision tokens dtype: {vision_tokens.dtype}")
        
        # Method 2: Encode from PIL Image
        img = Image.open(image_path).convert("RGB")
        vision_tokens_2 = encoder(img)  # Using __call__ shorthand
        
        # Verify results are the same
        assert torch.allclose(vision_tokens, vision_tokens_2, atol=1e-5)
        print("✓ Both encoding methods produce identical results")
        
    else:
        print(f"\nWarning: Example image not found at {image_path}")
        print("Please provide a valid image path to test encoding.")
    
    return encoder


def main():
    """Run both usage examples."""
    # Example 1: Simple one-line initialization (Recommended for most users)
    encoder_simple = simple_usage_example()
    
    # Example 2: Advanced manual loading (For users who need more control)
    encoder_advanced = advanced_usage_example()
    
    print("\n" + "=" * 60)
    print("Both methods work! Use from_pretrained() for simplicity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
