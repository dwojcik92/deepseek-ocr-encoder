"""Tests for DeepSeekOCREncoder."""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from deepseek_ocr_encoder import DeepSeekOCREncoder


class TestDeepSeekOCREncoder:
    """Test suite for DeepSeekOCREncoder."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock DeepSeek-OCR model."""
        mock = MagicMock()
        
        # Mock base_model structure
        base = MagicMock()
        base.sam_model = MagicMock()
        base.sam_model.return_value = torch.randn(1, 1024, 16, 16)
        
        vision = MagicMock()
        vision.pre_layrnorm = MagicMock()
        vision.transformer = MagicMock()
        vision.transformer.return_value = torch.randn(1, 256, 1024)
        
        # Mock position embeddings
        pos_weight = torch.randn(257, 1024)
        vision.embeddings.position_embedding.weight = pos_weight
        
        base.vision_model = vision
        mock.base_model = base
        
        return mock

    def test_encoder_initialization(self, mock_model):
        """Test encoder initialization."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            freeze=True,
            eager_to_device=False,
        )
        
        assert encoder.device.type == "cpu"
        assert encoder.dtype == torch.float32
        assert encoder.embed_dim == 1024

    def test_encoder_output_shape(self, mock_model):
        """Test that encoder output has correct shape."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            eager_to_device=False,
        )
        
        # Note: Actual testing would require a real image
        # This is a placeholder for structure validation
        assert hasattr(encoder, "encode")
        assert hasattr(encoder, "capture_cudagraph")

    def test_encoder_callable(self, mock_model):
        """Test that encoder is callable."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            eager_to_device=False,
        )
        
        assert callable(encoder)


if __name__ == "__main__":
    pytest.main([__file__])
