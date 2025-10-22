# Setup Guide

This guide will help you set up the DeepSeek OCR Encoder package using `uv`.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for optimal performance)
- `uv` package manager

## Installation Steps

### 1. Install uv (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Install the package in development mode

Navigate to the project directory and install with dependencies:

```bash
# Install core dependencies
uv pip install -e .

# Or install with dev dependencies for development
uv pip install -e ".[dev]"
```

### 3. Verify installation

```python
python -c "from deepseek_ocr_encoder import DeepSeekOCREncoder; print('✓ Installation successful!')"
```

## Using uv add for dependency management

If you want to add new dependencies to the project:

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add a dependency with version constraint
uv add "package-name>=1.0.0,<2.0.0"
```

## Quick Start

After installation, you can use the encoder as follows:

```python
from transformers import AutoModel
import torch
from deepseek_ocr_encoder import DeepSeekOCREncoder

# Load model
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Create encoder
encoder = DeepSeekOCREncoder(model, device="cuda")

# Encode image
tokens = encoder.encode("path/to/image.png")
print(f"Encoded tokens shape: {tokens.shape}")
```

## Development Workflow

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black src/ tests/ examples/
```

### Linting

```bash
ruff check src/ tests/ examples/
```

### Type checking

```bash
mypy src/
```

## Troubleshooting

### ImportError: No module named 'deepseek_ocr_encoder'

Make sure you installed the package in editable mode:
```bash
uv pip install -e .
```

### CUDA out of memory

Try using CPU or reducing batch size:
```python
encoder = DeepSeekOCREncoder(model, device="cpu")
```

### Missing dependencies

Reinstall with:
```bash
uv pip install -e . --force-reinstall
```
