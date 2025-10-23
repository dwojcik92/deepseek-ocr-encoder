"""
Example: Custom Preprocessing Hooks

This example demonstrates the new configurable preprocessing features that allow:
1. Custom preprocessing transforms
2. Configurable resize and normalization parameters
3. Native resolution support
4. Domain-specific preprocessing pipelines

These features enable users to customize the preprocessing pipeline without forking the code.
"""

import torch
from PIL import Image
from torchvision import transforms
from deepseek_ocr_encoder import DeepSeekOCREncoder


# Example 1: Custom resize dimensions (e.g., keep native resolution or use different size)
print("Example 1: Custom resize dimensions")
print("=" * 60)

# Use 512x512 instead of default 1024x1024
encoder_512 = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=(512, 512),
    device="cuda"
)

# Keep native resolution (no resize)
encoder_native = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=None,  # No resizing
    device="cuda"
)

# Use non-square dimensions
encoder_rect = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=(768, 1024),  # (height, width)
    device="cuda"
)

print("✓ Created encoders with custom resize dimensions")
print()


# Example 2: Custom normalization parameters (e.g., ImageNet normalization)
print("Example 2: Custom normalization parameters")
print("=" * 60)

# Use ImageNet normalization instead of CLIP
encoder_imagenet = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    normalization_mean=(0.485, 0.456, 0.406),  # ImageNet mean
    normalization_std=(0.229, 0.224, 0.225),   # ImageNet std
    device="cuda"
)

print("✓ Created encoder with ImageNet normalization")
print()


# Example 3: Custom preprocessing transform (domain-specific pipeline)
print("Example 3: Custom preprocessing transform")
print("=" * 60)

# Define a custom preprocessing pipeline for medical images
def medical_image_preprocessing(img: Image.Image) -> torch.Tensor:
    """Custom preprocessing for medical images with contrast enhancement."""
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ColorJitter(brightness=0.1, contrast=0.2),  # Enhance contrast
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),  # Custom normalization for grayscale/medical
            std=(0.5, 0.5, 0.5)
        ),
    ])
    return transform(img)


encoder_medical = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    preprocessing_transform=medical_image_preprocessing,
    device="cuda"
)

print("✓ Created encoder with custom medical image preprocessing")
print()


# Example 4: Reuse existing preprocessing pipeline
print("Example 4: Reuse existing preprocessing pipeline")
print("=" * 60)

# Suppose you have an existing preprocessing pipeline from your project
existing_pipeline = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

encoder_reuse = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    preprocessing_transform=existing_pipeline,
    device="cuda"
)

print("✓ Created encoder reusing existing preprocessing pipeline")
print()


# Example 5: Pre-processed tensor input (advanced use case)
print("Example 5: Pre-processed tensor input")
print("=" * 60)

# Create encoder that accepts pre-processed tensors
encoder_preprocessed = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    skip_default_preprocessing=True,  # No preprocessing applied
    device="cuda"
)

# Prepare your own preprocessed tensor
img = Image.open("example.jpg").convert("RGB")
custom_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])
preprocessed_tensor = custom_transform(img)  # [3, 1024, 1024]

# Encode the preprocessed tensor directly
tokens = encoder_preprocessed._encode_single_image(preprocessed_tensor)

print("✓ Encoded pre-processed tensor directly")
print(f"  Output shape: {tokens.shape}")
print()


# Example 6: Different interpolation modes
print("Example 6: Different interpolation modes")
print("=" * 60)

# Use LANCZOS for higher quality (but slower)
encoder_lanczos = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_interpolation=transforms.InterpolationMode.LANCZOS,
    device="cuda"
)

# Use BILINEAR for faster (but lower quality)
encoder_bilinear = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_interpolation=transforms.InterpolationMode.BILINEAR,
    resize_antialias=False,  # Disable antialiasing for speed
    device="cuda"
)

print("✓ Created encoders with different interpolation modes")
print()


# Example 7: Practical use case - Document processing pipeline
print("Example 7: Practical use case - Document processing")
print("=" * 60)

def document_preprocessing(img: Image.Image) -> torch.Tensor:
    """
    Specialized preprocessing for document images:
    - Convert to grayscale and back to RGB (simulate document scanning)
    - Apply sharpening filter
    - Standard resize and normalize
    """
    # Convert to grayscale and back (simulates scanned document)
    img_gray = img.convert('L').convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC),
        # Sharpen for better text recognition
        transforms.Lambda(lambda x: x),  # Placeholder for custom sharpening
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    return transform(img_gray)


encoder_document = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    preprocessing_transform=document_preprocessing,
    device="cuda"
)

print("✓ Created encoder with document-specific preprocessing")
print()


# Example 8: Batch processing with consistent preprocessing
print("Example 8: Batch processing with consistent preprocessing")
print("=" * 60)

# Create encoder with specific preprocessing for batch processing
encoder_batch = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=(1024, 1024),
    normalization_mean=(0.48145466, 0.4578275, 0.40821073),
    normalization_std=(0.26862954, 0.26130258, 0.27577711),
    device="cuda"
)

# Process multiple images with consistent preprocessing
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Note: This is conceptual - actual batch processing would require additional code
print("✓ Encoder ready for batch processing with consistent preprocessing")
print()


print("=" * 60)
print("Summary:")
print("The configurable preprocessing hooks allow you to:")
print("  • Customize resize dimensions (including native resolution)")
print("  • Configure normalization parameters")
print("  • Inject custom preprocessing transforms")
print("  • Reuse existing domain-specific pipelines")
print("  • Process pre-preprocessed tensors")
print("  • Fine-tune interpolation and quality settings")
print()
print("All of this without forking the codebase!")
