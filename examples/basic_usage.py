"""
Basic usage example for DeepSeek OCR Encoder.

This script demonstrates how to:
1. Load the DeepSeek-OCR model
2. Create an optimized encoder
3. Encode images to vision tokens
"""

import os
import torch
from transformers import AutoModel
from PIL import Image
from deepseek_ocr_encoder import DeepSeekOCREncoder


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the base DeepSeek-OCR model
    print("Loading DeepSeek-OCR model...")
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = model.eval().to(device, dtype=torch.bfloat16)
    print("Model loaded successfully!")

    # Create the optimized encoder
    print("\nCreating optimized encoder...")
    encoder = DeepSeekOCREncoder(
        full_model=model,
        device=device,
        dtype=torch.bfloat16,
        freeze=True,
        eager_to_device=True,
        precompute_pos_for_1024=True,
        use_compile=False,  # Set True for PyTorch 2.3+ with extra fusion
    )
    print("Encoder created!")

    # Optional: Capture CUDA graph for faster inference (CUDA only)
    if device == "cuda":
        print("\nCapturing CUDA graph for optimized inference...")
        encoder.capture_cudagraph(batch_size=1, H=1024, W=1024)
        print("CUDA graph captured!")

    # Example: Encode an image
    image_path = "example_image.png"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"\nEncoding image: {image_path}")
        
        # Method 1: Encode from path
        vision_tokens = encoder.encode(image_path)
        print(f"Vision tokens shape: {vision_tokens.shape}")
        print(f"Vision tokens dtype: {vision_tokens.dtype}")
        
        # Method 2: Encode from PIL Image
        img = Image.open(image_path).convert("RGB")
        vision_tokens_2 = encoder(img)  # Using __call__ shorthand
        
        # Verify results are the same
        assert torch.allclose(vision_tokens, vision_tokens_2, atol=1e-5)
        print("✓ Both encoding methods produce identical results")
        
    else:
        print(f"\nWarning: Example image not found at {image_path}")
        print("Please provide a valid image path to test encoding.")


if __name__ == "__main__":
    main()
