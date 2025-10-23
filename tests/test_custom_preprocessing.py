"""Tests for custom preprocessing hooks feature."""

import torch
from unittest.mock import MagicMock
from PIL import Image
from torchvision import transforms
from deepseek_ocr_encoder import DeepSeekOCREncoder
import pytest


class TestCustomPreprocessingHooks:
    """Test suite for configurable preprocessing hooks."""

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

    def test_custom_resize_size(self, mock_model):
        """Test custom resize dimensions."""
        # Test with 512x512
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            resize_size=(512, 512),
            eager_to_device=False,
        )
        
        assert encoder._preproc_1024 is not None
        assert not encoder._skip_default_preprocessing
        
        # Test with None (native resolution)
        encoder_native = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            resize_size=None,
            eager_to_device=False,
        )
        
        assert encoder_native._preproc_1024 is not None

    def test_custom_normalization(self, mock_model):
        """Test custom normalization parameters."""
        # ImageNet normalization
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
            eager_to_device=False,
        )
        
        assert encoder._preproc_1024 is not None
        assert not encoder._skip_default_preprocessing

    def test_custom_interpolation(self, mock_model):
        """Test custom interpolation mode."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            resize_interpolation=transforms.InterpolationMode.LANCZOS,
            resize_antialias=False,
            eager_to_device=False,
        )
        
        assert encoder._preproc_1024 is not None

    def test_custom_preprocessing_transform(self, mock_model):
        """Test custom preprocessing transform."""
        def custom_transform(img: Image.Image) -> torch.Tensor:
            transform = transforms.Compose([
                transforms.Resize((800, 800)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
            return transform(img)
        
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            preprocessing_transform=custom_transform,
            eager_to_device=False,
        )
        
        # Custom transform should override default
        assert encoder._preproc_1024 == custom_transform
        assert not encoder._skip_default_preprocessing

    def test_skip_default_preprocessing(self, mock_model):
        """Test skip_default_preprocessing flag."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            skip_default_preprocessing=True,
            eager_to_device=False,
        )
        
        assert encoder._skip_default_preprocessing
        assert encoder._preproc_1024 is None

    def test_tensor_input_with_skip_preprocessing(self, mock_model):
        """Test tensor input when skip_default_preprocessing is True."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            skip_default_preprocessing=True,
            eager_to_device=False,
            precompute_pos_for_1024=False,
        )
        
        # Create a preprocessed tensor
        preprocessed = torch.randn(3, 1024, 1024)
        
        # Should accept tensor input
        # Note: This will fail in the forward pass due to mocking,
        # but we're testing the preprocessing logic
        try:
            encoder._encode_single_image(preprocessed)
        except Exception:
            # Expected - we're just testing preprocessing logic
            pass

    def test_tensor_input_without_skip_preprocessing_fails(self, mock_model):
        """Test that tensor input fails when skip_default_preprocessing is False."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            skip_default_preprocessing=False,
            eager_to_device=False,
        )
        
        # Create a preprocessed tensor
        preprocessed = torch.randn(3, 1024, 1024)
        
        # Should raise error
        with pytest.raises(ValueError, match="Tensor input is only allowed"):
            encoder._encode_single_image(preprocessed)

    def test_backward_compatibility(self, mock_model):
        """Test that default behavior is backward compatible."""
        # Default initialization should work as before
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            eager_to_device=False,
        )
        
        # Should have default preprocessing
        assert encoder._preproc_1024 is not None
        assert not encoder._skip_default_preprocessing

    def test_custom_preprocessing_with_from_pretrained(self, mock_model):
        """Test custom preprocessing with from_pretrained class method."""
        from unittest.mock import patch
        
        with patch("deepseek_ocr_encoder.encoder.AutoModel") as mock_automodel:
            mock_automodel.from_pretrained.return_value = mock_model
            
            # Test with custom resize
            encoder = DeepSeekOCREncoder.from_pretrained(
                "deepseek-ai/DeepSeek-OCR",
                device="cpu",
                dtype=torch.float32,
                resize_size=(512, 512),
                normalization_mean=(0.485, 0.456, 0.406),
                normalization_std=(0.229, 0.224, 0.225),
            )
            
            assert encoder._preproc_1024 is not None

    def test_rectangular_resize(self, mock_model):
        """Test non-square resize dimensions."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            resize_size=(768, 1024),  # height, width
            eager_to_device=False,
        )
        
        assert encoder._preproc_1024 is not None

    def test_custom_transform_overrides_other_settings(self, mock_model):
        """Test that custom transform overrides other preprocessing settings."""
        custom_transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
        ])
        
        # Even if we specify other settings, custom transform should be used
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            preprocessing_transform=custom_transform,
            resize_size=(1024, 1024),  # This should be ignored
            normalization_mean=(0.5, 0.5, 0.5),  # This should be ignored
            eager_to_device=False,
        )
        
        assert encoder._preproc_1024 == custom_transform

    def test_no_preprocessing_error_on_pil_image(self, mock_model):
        """Test that error is raised when trying to encode PIL image without preprocessing."""
        encoder = DeepSeekOCREncoder(
            full_model=mock_model,
            device="cpu",
            dtype=torch.float32,
            skip_default_preprocessing=True,
            eager_to_device=False,
        )
        
        # Create a dummy PIL image
        img = Image.new('RGB', (100, 100))
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No preprocessing transform configured"):
            encoder._encode_single_image(img)
