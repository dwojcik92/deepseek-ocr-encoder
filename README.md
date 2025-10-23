# DeepSeek OCR Encoder

A handy and elastic encoder for vision tasks based on DeepSeek-OCR. This package provides an optimized, memory-lean encoder that combines SAM-base with CLIP for efficient vision token generation.

## Features

- 🚀 **Optimized Performance**: Leverages CUDA graphs, torch.compile, and memory-efficient techniques
- 💾 **Memory Efficient**: Automatically removes unused model components to save RAM/VRAM
- 🎯 **Easy to Use**: Simple API - just import and encode
- ⚡ **Fast Inference**: Support for BF16, channels_last memory layout, and optional CUDA graph capture
- 🔧 **Flexible**: Configurable device, dtype, and optimization settings
- 📄 **PDF Support**: Encode multi-page PDF documents with automatic page-to-image conversion

## About DeepSeek-OCR

This encoder is based on [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), a state-of-the-art vision-language model designed for optical character recognition and document understanding. The recent paper ["DeepSeek-OCR: Contexts Optical Compression"](https://arxiv.org/html/2510.18234v1) (arXiv:2510.18234v1) introduces innovative optical compression techniques for long text contexts using vision tokens.

**Key highlights from the paper:**
- 📊 **High Precision OCR**: Achieves up to ~97% OCR precision at less than 10× compression
- 🗜️ **Efficient Compression**: Maintains ~60% precision even at 20× compression ratios
- 📈 **Strong Benchmark Results**: Significant improvements on OmniDocBench
- ⚡ **High-Throughput Data Generation**: Enables efficient processing of large document datasets

This encoder package provides an optimized implementation for extracting vision tokens from the DeepSeek-OCR model, making it easy to integrate into your own applications.

## Installation

```bash
uv add deepseek-ocr-encoder
```

Or install from source:

```bash
git clone https://github.com/dwojcik92/deepseek-ocr-encoder.git
cd deepseek-ocr-encoder
uv pip install .
```

**Important:** This package requires `transformers>=4.30.0,<4.48.0`. If you have a newer version already installed, you may need to downgrade:

```bash
uv pip install 'transformers>=4.30.0,<4.48.0'
```

## Quick Start

### Simple One-Line Initialization (Recommended)

```python
from deepseek_ocr_encoder import DeepSeekOCREncoder

# One-line initialization - automatically handles device, dtype, and model loading
encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")

# Encode an image
vision_tokens = encoder("your_image.png")
# Returns: torch.Tensor of shape [1, N, 1024] where N=256 for 1024x1024 input
```

### Advanced Usage with Manual Model Loading

If you need more control over the model loading process:

```python
from transformers import AutoModel
import torch
from deepseek_ocr_encoder import DeepSeekOCREncoder
from PIL import Image

# Load the base DeepSeek-OCR model
model_name = "deepseek-ai/DeepSeek-OCR"
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model = model.eval().to("cuda", dtype=torch.bfloat16)

# Create the optimized encoder
encoder = DeepSeekOCREncoder(
    full_model=model,
    device="cuda",
    dtype=torch.bfloat16,
    freeze=True,
    eager_to_device=True,
    precompute_pos_for_1024=True,
    use_compile=False,  # Set True for PyTorch 2.3+ with extra fusion
)

# Optional: Capture CUDA graph for even faster inference
encoder.capture_cudagraph(batch_size=1, H=1024, W=1024)

# Encode an image
image_path = "your_image.png"
vision_tokens = encoder.encode(image_path)
# Returns: torch.Tensor of shape [1, N, 1024] where N=256 for 1024x1024 input

# Or use with PIL Image
img = Image.open(image_path).convert("RGB")
vision_tokens = encoder(img)  # Shorthand for encoder.encode(img)

# Encode a PDF document (multi-page support)
pdf_path = "document.pdf"
vision_tokens_list = encoder.encode(pdf_path)
# Returns: List of torch.Tensor, one per page, each of shape [1, N, 1024]

# Process each page
for page_num, page_tokens in enumerate(vision_tokens_list):
    print(f"Page {page_num + 1}: {page_tokens.shape}")
```

## API Reference

### DeepSeekOCREncoder

The main encoder class that wraps the DeepSeek-OCR model for efficient vision token extraction.

#### Class Methods

##### `from_pretrained(model_name_or_path: str, **kwargs) -> DeepSeekOCREncoder`

**(Recommended)** Load a DeepSeek-OCR model and wrap it with the optimized encoder in one line.

**Parameters:**
- `model_name_or_path` (str, required): Model identifier from Hugging Face Hub (e.g., "deepseek-ai/DeepSeek-OCR") or path to a local checkpoint
- `device` (Optional[Union[str, torch.device]]): Target device (default: auto-detect cuda if available, else cpu)
- `dtype` (Optional[torch.dtype]): Data type for computation (default: bfloat16 on cuda, float32 on cpu)
- `freeze` (bool): Whether to freeze encoder parameters (default: True)
- `eager_to_device` (bool): Move model to device immediately (default: True)
- `precompute_pos_for_1024` (bool): Pre-compute position embeddings for 1024x1024 input (default: True)
- `use_compile` (bool): Enable torch.compile for better performance (requires PyTorch 2.3+, default: False)
- `trust_remote_code` (bool): Whether to trust remote code when loading model (default: True)
- `use_safetensors` (bool): Whether to use safetensors format (default: True)
- `attn_implementation` (str): Attention implementation to use (default: "eager")
- `**model_kwargs`: Additional keyword arguments passed to AutoModel.from_pretrained()

**Returns:**
- Initialized `DeepSeekOCREncoder` ready for inference

**Example:**
```python
# Simple usage
encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")

# With custom device/dtype
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    device="cpu",
    dtype=torch.float32
)

# From local checkpoint
encoder = DeepSeekOCREncoder.from_pretrained("./my-finetuned-model")
```

#### Instance Methods

##### `encode(image: Union[Image.Image, str, os.PathLike]) -> Union[torch.Tensor, List[torch.Tensor]]`

Encode an image or PDF into vision tokens.

**Parameters:**
- `image`: PIL Image, path to an RGB image file, or path to a PDF file

**Returns:**
- For single images: Vision tokens tensor of shape `[1, N, 1024]` where N=256 for 1024×1024 input
- For PDFs: List of vision token tensors, one per page, each of shape `[1, N, 1024]`

**Example:**
```python
# Single image
tokens = encoder.encode("image.png")  # Returns torch.Tensor

# Multi-page PDF
tokens_list = encoder.encode("document.pdf")  # Returns List[torch.Tensor]
for page_tokens in tokens_list:
    print(f"Page shape: {page_tokens.shape}")
```

##### `capture_cudagraph(batch_size: int = 1, H: int = 1024, W: int = 1024)`

Capture a CUDA graph for optimized steady-state inference. Call this once after initialization to enable CUDA graph acceleration.

**Parameters:**
- `batch_size`: Batch size for the graph (default: 1)
- `H`: Input height (default: 1024)
- `W`: Input width (default: 1024)

**Raises:**
- `RuntimeError`: If device is not CUDA

##### `__call__(image: Union[Image.Image, str, os.PathLike]) -> Union[torch.Tensor, List[torch.Tensor]]`

Convenience method, equivalent to `encode()`. Supports both single images and multi-page PDFs.

## Custom Preprocessing Hooks

The encoder now supports configurable preprocessing, allowing you to customize the image preprocessing pipeline without forking the codebase. This is useful for:
- Using native image resolutions
- Applying domain-specific preprocessing (medical images, documents, etc.)
- Reusing existing preprocessing pipelines
- Fine-tuning preprocessing parameters

### Basic Examples

#### Custom Resize Dimensions

```python
# Use 512x512 instead of default 1024x1024
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=(512, 512)
)

# Keep native resolution (no resizing)
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=None
)

# Use non-square dimensions
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_size=(768, 1024)  # (height, width)
)
```

#### Custom Normalization

```python
# Use ImageNet normalization instead of CLIP
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225)
)
```

#### Custom Interpolation Mode

```python
from torchvision import transforms

# Use LANCZOS for higher quality
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    resize_interpolation=transforms.InterpolationMode.LANCZOS,
    resize_antialias=True
)
```

### Advanced: Custom Preprocessing Transform

For full control, provide your own preprocessing function:

```python
from torchvision import transforms
from PIL import Image
import torch

def my_preprocessing(img: Image.Image) -> torch.Tensor:
    """Custom preprocessing with domain-specific augmentations."""
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ColorJitter(brightness=0.1, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    return transform(img)

encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    preprocessing_transform=my_preprocessing
)
```

### Pre-processed Tensor Input

If you need to preprocess images externally (e.g., in a batched data pipeline):

```python
# Create encoder that accepts pre-processed tensors
encoder = DeepSeekOCREncoder.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    skip_default_preprocessing=True
)

# Your external preprocessing
img = Image.open("image.jpg").convert("RGB")
preprocessed = my_external_pipeline(img)  # Returns torch.Tensor [C, H, W]

# Encode the pre-processed tensor
tokens = encoder._encode_single_image(preprocessed)
```

### Preprocessing Parameters

When using `from_pretrained()` or the constructor, you can configure:

- `preprocessing_transform`: Custom callable that takes PIL Image and returns torch.Tensor (overrides all other settings)
- `resize_size`: Target size (int or tuple). Default: (1024, 1024). Set to None for native resolution
- `resize_interpolation`: Interpolation mode (default: `BICUBIC`)
- `resize_antialias`: Enable antialiasing during resize (default: True)
- `normalization_mean`: RGB mean values (default: CLIP normalization)
- `normalization_std`: RGB std values (default: CLIP normalization)
- `skip_default_preprocessing`: If True, accept only pre-processed tensors (default: False)

See `examples/custom_preprocessing.py` for more detailed examples.

## Architecture

The encoder implements the following pipeline:

1. **SAM-base encoder** with built-in conv compressor → `[B, 1024, Hs, Ws]`
2. **Flatten** spatial dimensions → `[B, N, 1024]` where N = Hs × Ws
3. **Add CLIP 2D positional embeddings** (without CLS token)
4. **CLIP pre-layernorm + transformer**
5. **Residual connection**: returns `tokens + CLIP(tokens)`

## Performance Optimizations

This encoder includes several optimizations:

- **Memory layout**: Uses `channels_last` format for conv-heavy operations
- **Precision**: BF16 computation for faster inference on modern GPUs
- **CUDA Graphs**: Optional graph capture for minimal kernel launch overhead
- **torch.compile**: Optional compilation for kernel fusion (PyTorch 2.3+)
- **Memory cleanup**: Removes unused model components (text decoder, LM head, etc.)
- **Position embedding caching**: Pre-computes and caches position embeddings

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- **transformers ≥ 4.30.0, < 4.48.0** (see [Troubleshooting](#troubleshooting) for details)
- Pillow ≥ 9.0.0
- PyMuPDF ≥ 1.23.0 (for PDF support)

## Troubleshooting

### ImportError: cannot import name 'LlamaFlashAttention2'

If you encounter this error, it's caused by incompatible transformers versions. The `LlamaFlashAttention2` class was removed in transformers 4.48.0+.

**Solution:**
```bash
uv pip install 'transformers>=4.30.0,<4.48.0'
```

The DeepSeek-OCR model uses specific attention mechanisms that were refactored in transformers 4.48.0+. The model code references `LlamaFlashAttention2`, which is only available in transformers versions 4.30.0 through 4.47.x.

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT

## Citation

If you use this encoder in your research, please cite the DeepSeek-OCR papers:

```bibtex
@article{deepseek-ocr-compression,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## Resources

- 📄 **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/html/2510.18234v1) (arXiv:2510.18234v1)
- 💻 **Official Repository**: [DeepSeek-OCR on GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- 🤗 **Model**: [deepseek-ai/DeepSeek-OCR on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
