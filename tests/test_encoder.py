"""Tests for DeepSeekOCREncoder."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
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

    @patch("deepseek_ocr_encoder.encoder.AutoModel")
    def test_from_pretrained_basic(self, mock_automodel, mock_model):
        """Test from_pretrained() basic functionality."""
        # Mock AutoModel.from_pretrained to return our mock model
        mock_automodel.from_pretrained.return_value = mock_model
        
        # Call from_pretrained
        encoder = DeepSeekOCREncoder.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            device="cpu",
            dtype=torch.float32,
        )
        
        # Verify AutoModel.from_pretrained was called with correct args
        mock_automodel.from_pretrained.assert_called_once()
        call_args = mock_automodel.from_pretrained.call_args
        assert call_args[0][0] == "deepseek-ai/DeepSeek-OCR"
        assert call_args[1]["trust_remote_code"] == True
        assert call_args[1]["use_safetensors"] == True
        assert call_args[1]["torch_dtype"] == torch.float32
        assert call_args[1]["attn_implementation"] == "eager"
        
        # Verify encoder was created properly
        assert encoder.device.type == "cpu"
        assert encoder.dtype == torch.float32

    @patch("deepseek_ocr_encoder.encoder.AutoModel")
    @patch("torch.cuda.is_available")
    def test_from_pretrained_auto_device_cuda(self, mock_cuda_available, mock_automodel, mock_model):
        """Test from_pretrained() auto-detects CUDA device."""
        mock_cuda_available.return_value = True
        mock_automodel.from_pretrained.return_value = mock_model
        
        encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")
        
        # Should auto-detect cuda device
        assert encoder.device.type == "cuda"
        # Should default to bfloat16 on cuda
        assert encoder.dtype == torch.bfloat16

    @patch("deepseek_ocr_encoder.encoder.AutoModel")
    @patch("torch.cuda.is_available")
    def test_from_pretrained_auto_device_cpu(self, mock_cuda_available, mock_automodel, mock_model):
        """Test from_pretrained() auto-detects CPU device when CUDA unavailable."""
        mock_cuda_available.return_value = False
        mock_automodel.from_pretrained.return_value = mock_model
        
        encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")
        
        # Should auto-detect cpu device
        assert encoder.device.type == "cpu"
        # Should default to float32 on cpu
        assert encoder.dtype == torch.float32

    @patch("deepseek_ocr_encoder.encoder.AutoModel")
    def test_from_pretrained_local_path(self, mock_automodel, mock_model):
        """Test from_pretrained() works with local model path."""
        mock_automodel.from_pretrained.return_value = mock_model
        
        encoder = DeepSeekOCREncoder.from_pretrained(
            "./my-local-model",
            device="cpu",
            dtype=torch.float32,
        )
        
        # Verify AutoModel.from_pretrained was called with local path
        call_args = mock_automodel.from_pretrained.call_args
        assert call_args[0][0] == "./my-local-model"
        
        assert encoder.device.type == "cpu"

    @patch("deepseek_ocr_encoder.encoder.AutoModel")
    def test_from_pretrained_custom_kwargs(self, mock_automodel, mock_model):
        """Test from_pretrained() passes through custom model kwargs."""
        mock_automodel.from_pretrained.return_value = mock_model
        
        encoder = DeepSeekOCREncoder.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            device="cpu",
            dtype=torch.float32,
            low_cpu_mem_usage=True,  # Custom kwarg
            revision="main",  # Custom kwarg
        )
        
        # Verify custom kwargs were passed through
        call_args = mock_automodel.from_pretrained.call_args
        assert call_args[1]["low_cpu_mem_usage"] == True
        assert call_args[1]["revision"] == "main"


if __name__ == "__main__":
    pytest.main([__file__])
