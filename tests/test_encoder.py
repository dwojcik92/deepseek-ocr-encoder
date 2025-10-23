"""Tests for DeepSeekOCREncoder."""

import io
import os
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
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
        
        # Mock position embeddings - need actual tensor that can be registered as buffer
        pos_weight = torch.randn(257, 1024)
        mock_weight = MagicMock()
        mock_weight.detach.return_value = pos_weight
        vision.embeddings.position_embedding.weight = mock_weight
        
        base.vision_model = vision
        mock.base_model = base
        
        # Make eval() and to() return self to support chaining
        mock.eval.return_value = mock
        mock.to.return_value = mock
        
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
        assert call_args[1]["trust_remote_code"]
        assert call_args[1]["use_safetensors"]
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
        )
        
        # Verify it was called with the local path
        call_args = mock_automodel.from_pretrained.call_args
        assert call_args[0][0] == "./my-local-model"
        assert encoder is not None

    def test_is_pdf_detection(self):
        """Test PDF file detection."""
        assert DeepSeekOCREncoder._is_pdf("document.pdf") is True
        assert DeepSeekOCREncoder._is_pdf("Document.PDF") is True
        assert DeepSeekOCREncoder._is_pdf("/path/to/file.pdf") is True
        assert DeepSeekOCREncoder._is_pdf("image.png") is False
        assert DeepSeekOCREncoder._is_pdf("image.jpg") is False
        assert DeepSeekOCREncoder._is_pdf("file.txt") is False

    def test_pdf_to_images_requires_pymupdf(self, tmp_path):
        """Test that PDF conversion requires PyMuPDF."""
        # Create a dummy PDF path (doesn't need to exist for this test)
        pdf_path = tmp_path / "test.pdf"
        
        # Mock the HAS_PYMUPDF flag
        with patch('deepseek_ocr_encoder.encoder.HAS_PYMUPDF', False):
            with pytest.raises(ImportError, match="PyMuPDF is required"):
                DeepSeekOCREncoder._pdf_to_images(pdf_path)

    @patch('deepseek_ocr_encoder.encoder.HAS_PYMUPDF', True)
    @patch('deepseek_ocr_encoder.encoder.fitz')
    def test_pdf_to_images_conversion(self, mock_fitz, tmp_path):
        """Test PDF to images conversion."""
        # Create mock PDF document with 2 pages
        mock_pdf = MagicMock()
        
        # Mock pages
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        
        # Create simple test images as bytes
        test_img1 = Image.new('RGB', (100, 100), color='red')
        test_img2 = Image.new('RGB', (100, 100), color='blue')
        
        img1_bytes = io.BytesIO()
        img2_bytes = io.BytesIO()
        test_img1.save(img1_bytes, format='PNG')
        test_img2.save(img2_bytes, format='PNG')
        
        # Mock pixmaps
        mock_pix1 = MagicMock()
        mock_pix1.tobytes.return_value = img1_bytes.getvalue()
        mock_pix2 = MagicMock()
        mock_pix2.tobytes.return_value = img2_bytes.getvalue()
        
        mock_page1.get_pixmap.return_value = mock_pix1
        mock_page2.get_pixmap.return_value = mock_pix2
        
        # Set up iteration over PDF pages
        mock_pdf.__iter__.return_value = iter([mock_page1, mock_page2])
        mock_fitz.open.return_value = mock_pdf
        mock_fitz.Matrix.return_value = MagicMock()
        
        # Test conversion
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()  # Create empty file
        
        images = DeepSeekOCREncoder._pdf_to_images(pdf_path)
        
        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)
        assert mock_pdf.close.called

    @patch('deepseek_ocr_encoder.encoder.HAS_PYMUPDF', True)
    def test_encode_returns_list_for_pdf(self, mock_model, tmp_path):
        """Test that encode returns a list for PDF input."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            eager_to_device=False,
        )
        
        # Create a fake PDF path (file doesn't need to exist since _pdf_to_images is mocked)
        pdf_path = tmp_path / "test.pdf"
        
        # Mock the PDF conversion to return test images
        test_images = [Image.new('RGB', (100, 100)) for _ in range(3)]
        
        with patch.object(DeepSeekOCREncoder, '_pdf_to_images', return_value=test_images):
            with patch.object(encoder, '_encode_single_image', return_value=torch.randn(1, 256, 1024)):
                result = encoder.encode(str(pdf_path))
                
                assert isinstance(result, list)
                assert len(result) == 3
                assert all(isinstance(t, torch.Tensor) for t in result)


if __name__ == "__main__":
    pytest.main([__file__])
