"""DeepSeek OCR Encoder - A handy and elastic encoder for vision tasks."""

import sys

# Version validation for transformers compatibility
def _check_transformers_version():
    """Check if transformers version is compatible."""
    try:
        import transformers
        from packaging import version
    except ImportError:
        # If packaging is not available, skip version check
        return
    
    transformers_version = transformers.__version__
    try:
        parsed_version = version.parse(transformers_version)
        min_version = version.parse("4.30.0")
        max_version = version.parse("4.48.0")
        
        if parsed_version < min_version or parsed_version >= max_version:
            print(
                f"\n{'='*70}\n"
                f"WARNING: Incompatible transformers version detected!\n"
                f"{'='*70}\n"
                f"Current version: {transformers_version}\n"
                f"Required version: >=4.30.0,<4.48.0\n\n"
                f"The DeepSeek-OCR model requires specific transformers features that\n"
                f"may not be available in version {transformers_version}.\n\n"
                f"To fix this issue, please run:\n"
                f"  pip install 'transformers>=4.30.0,<4.48.0'\n\n"
                f"Recommended version: transformers==4.47.0\n"
                f"{'='*70}\n",
                file=sys.stderr
            )
    except Exception:
        # If version parsing fails, continue silently
        pass

_check_transformers_version()

from .encoder import DeepSeekOCREncoder

__version__ = "0.2.0"
__all__ = ["DeepSeekOCREncoder"]
