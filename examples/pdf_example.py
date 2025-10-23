"""
PDF encoding example for DeepSeek OCR Encoder.

This script demonstrates how to:
1. Load the DeepSeek-OCR model
2. Create an optimized encoder
3. Encode multi-page PDF documents to vision tokens
"""

import os
import torch
from transformers import AutoModel
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
        use_compile=False,
    )
    print("Encoder created!")

    # Optional: Capture CUDA graph for faster inference (CUDA only)
    if device == "cuda":
        print("\nCapturing CUDA graph for optimized inference...")
        encoder.capture_cudagraph(batch_size=1, H=1024, W=1024)
        print("CUDA graph captured!")

    # Example 1: Encode a PDF file
    pdf_path = "document.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print(f"\nEncoding PDF: {pdf_path}")
        
        # Encode PDF - returns a list of tensors, one per page
        vision_tokens_list = encoder.encode(pdf_path)
        
        print(f"Number of pages: {len(vision_tokens_list)}")
        for i, tokens in enumerate(vision_tokens_list):
            print(f"Page {i+1} - Vision tokens shape: {tokens.shape}, dtype: {tokens.dtype}")
        
        # You can also use the __call__ shorthand
        vision_tokens_list_2 = encoder(pdf_path)
        
        # Verify results are the same
        for i, (t1, t2) in enumerate(zip(vision_tokens_list, vision_tokens_list_2)):
            assert torch.allclose(t1, t2, atol=1e-5)
        print("✓ Both encoding methods produce identical results")
        
        # Example: Process each page individually
        print("\nProcessing each page:")
        for i, page_tokens in enumerate(vision_tokens_list):
            # Here you can do further processing with each page's tokens
            # For example, pass to a language model, store in a database, etc.
            print(f"  Page {i+1}: {page_tokens.shape[1]} tokens of dimension {page_tokens.shape[2]}")
        
    else:
        print(f"\nWarning: Example PDF not found at {pdf_path}")
        print("Please provide a valid PDF path to test encoding.")
        print("\nYou can create a test PDF or use any existing PDF document.")


if __name__ == "__main__":
    main()
