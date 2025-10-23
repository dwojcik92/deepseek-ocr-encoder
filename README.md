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

### Using uv (recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd deepseek-ocr-encoder

# Install with uv
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

**Important:** This package requires `transformers>=4.30.0,<4.48.0`. The installation will automatically install a compatible version. If you have a newer version of transformers already installed, you may need to downgrade:

```bash
pip install 'transformers>=4.30.0,<4.48.0'
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

#### Constructor Parameters

- `full_model` (required): The full DeepSeek-OCR model loaded from transformers
- `device` (Optional[Union[str, torch.device]]): Target device (default: cuda if available, else cpu)
- `dtype` (torch.dtype): Data type for computation (default: torch.bfloat16)
- `freeze` (bool): Whether to freeze encoder parameters (default: True)
- `eager_to_device` (bool): Move model to device immediately (default: True)
- `precompute_pos_for_1024` (bool): Pre-compute position embeddings for 1024x1024 input (default: True)
- `use_compile` (bool): Enable torch.compile for better performance (requires PyTorch 2.3+)

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

> **Note:** The package requires transformers version < 4.48.0 due to API changes in newer versions. We recommend using transformers==4.47.0 for optimal compatibility.

## Troubleshooting

### ImportError: cannot import name 'LlamaFlashAttention2'

If you encounter this error:
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This is caused by incompatible transformers versions. The `LlamaFlashAttention2` class was removed in transformers 4.48.0 and later versions.

**Solution:**

Install a compatible version of transformers:
```bash
pip install 'transformers>=4.30.0,<4.48.0'
```

We recommend using transformers==4.47.0:
```bash
pip install transformers==4.47.0
```

**Why this happens:**

The DeepSeek-OCR model uses specific attention mechanisms that were refactored in transformers 4.48.0+. The model code (loaded via `trust_remote_code=True`) references `LlamaFlashAttention2`, which is only available in transformers versions 4.30.0 through 4.47.x.

### Version Validation

When you import the package, it will automatically check your transformers version and display a warning if an incompatible version is detected:

```python
from deepseek_ocr_encoder import DeepSeekOCREncoder  # Warning displayed if version incompatible
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
ruff check src/
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

@article{deepseek-ocr,
  title={DeepSeek-OCR: Efficient Vision-Language Model for OCR},
  author={DeepSeek-AI},
  year={2024}
}
```

## Resources

- 📄 **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/html/2510.18234v1) (arXiv:2510.18234v1)
- 💻 **Official Repository**: [DeepSeek-OCR on GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- 🤗 **Model**: [deepseek-ai/DeepSeek-OCR on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
