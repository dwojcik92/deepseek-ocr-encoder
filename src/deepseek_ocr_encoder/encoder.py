"""
DeepSeek OCR Encoder

An optimized, memory-lean encoder that combines SAM-base with CLIP for vision token generation.
"""

import math
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


class DeepSeekOCREncoder(torch.nn.Module):
    """
    Optimized, memory-lean encoder:
      SAM-base (with built-in conv compressor) -> [B,1024,Hs,Ws]
      flatten -> add CLIP 2D pos-emb (no CLS)
      -> CLIP pre-layernorm + transformer
      returns tokens + CLIP(tokens)  [B, N, 1024]
    """

    def __init__(
        self,
        full_model,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        eager_to_device: bool = True,
        precompute_pos_for_1024: bool = True,
        use_compile: bool = False,
    ):
        """
        Initialize the DeepSeek OCR Encoder.

        Args:
            full_model: The full DeepSeek-OCR model loaded from transformers
            device: Target device (default: cuda if available, else cpu)
            dtype: Data type for computation (default: bfloat16)
            freeze: Whether to freeze encoder parameters (default: True)
            eager_to_device: Move model to device immediately (default: True)
            precompute_pos_for_1024: Pre-compute position embeddings for 1024x1024 input (default: True)
            use_compile: Enable torch.compile for better performance (requires PyTorch 2.3+)
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.embed_dim = 1024

        base = getattr(full_model, "base_model", full_model)
        self.sam = getattr(base, "sam_model").eval()
        vision = getattr(base, "vision_model")
        self.clip_pre = vision.pre_layrnorm.eval()
        self.clip_tr = vision.transformer.eval()

        # Position table (register as buffer so it moves with .to())
        pos_table = vision.embeddings.position_embedding.weight.detach()  # [1+G^2, 1024]
        self.register_buffer("clip_pos_table", pos_table, persistent=False)

        # Drop likely-unused heavy parts to save RAM/VRAM
        def _safe_del(obj, name):
            if hasattr(obj, name):
                try:
                    setattr(obj, name, None)
                except Exception:
                    pass

        _safe_del(vision.embeddings, "patch_embedding")
        _safe_del(vision, "post_layernorm")
        _safe_del(base, "text_model")
        _safe_del(base, "lm_head")
        _safe_del(full_model, "text_model")
        _safe_del(full_model, "lm_head")

        # If you want N=256 like the paper figure, feed 1024×1024.
        self._preproc_1024 = transforms.Compose([
            transforms.Resize(
                (1024, 1024),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.ToTensor(),
            # Use the SAM/CLIP-normalization your training used.
            # If unsure, CLIP's works well in practice:
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        if freeze:
            for m in [self.sam, self.clip_pre, self.clip_tr]:
                for p in m.parameters():
                    p.requires_grad_(False)

        if eager_to_device:
            # channels_last helps conv-heavy encoders; safe for BF16
            self.sam.to(self.device, dtype=self.dtype, memory_format=torch.channels_last)
            self.clip_pre.to(self.device, dtype=self.dtype)
            self.clip_tr.to(self.device, dtype=self.dtype)
            self.clip_pos_table.data = self.clip_pos_table.to(self.device, dtype=self.dtype)

        # Precompute pos-emb for the common 1024→N=256 case (Hs=Ws=16)
        self._pos_cache = {}
        self._pos_fixed_16 = None
        if precompute_pos_for_1024:
            table = self.clip_pos_table
            grid_size = int(math.isqrt(table.size(0) - 1))
            if grid_size * grid_size + 1 != table.size(0):
                raise RuntimeError("Position table size is not 1+G^2; cannot infer grid size.")
            base_grid = table[1: 1 + grid_size * grid_size]  # [G^2, 1024]
            base_grid = base_grid.view(grid_size, grid_size, self.embed_dim).permute(2, 0, 1).unsqueeze(0)
            if grid_size != 16:
                base_grid = F.interpolate(base_grid, size=(16, 16), mode="bicubic", align_corners=False)
            self._pos_fixed_16 = base_grid.flatten(2).transpose(1, 2).contiguous()  # [1,256,1024]

        # Optional torch.compile for better kernel fusion on the transformer path
        self._compiled = None
        if use_compile and hasattr(torch, "compile"):
            self._compiled = torch.compile(self._forward_core, mode="max-autotune")  # PyTorch 2.3+

        # Optional CUDA Graphs (opt-in; call capture_cudagraph() once before steady-state)
        self._graph = None
        self._static_in = None
        self._static_out = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
        eager_to_device: bool = True,
        precompute_pos_for_1024: bool = True,
        use_compile: bool = False,
        trust_remote_code: bool = True,
        use_safetensors: bool = True,
        attn_implementation: str = "eager",
        **model_kwargs,
    ):
        """
        Load a DeepSeek-OCR model and wrap it with the optimized encoder.

        This is a convenience method that handles the full initialization pipeline:
        - Downloads/loads the DeepSeek-OCR model from Hugging Face or local path
        - Automatically detects CUDA/CPU and sets the device
        - Sets recommended defaults (bfloat16 on CUDA, float32 on CPU)
        - Wraps the model with the optimized encoder

        Args:
            model_name_or_path: Model identifier from Hugging Face Hub (e.g., "deepseek-ai/DeepSeek-OCR")
                or path to a local checkpoint directory
            device: Target device (default: auto-detect cuda if available, else cpu)
            dtype: Data type for computation (default: bfloat16 on cuda, float32 on cpu)
            freeze: Whether to freeze encoder parameters (default: True)
            eager_to_device: Move model to device immediately (default: True)
            precompute_pos_for_1024: Pre-compute position embeddings for 1024x1024 input (default: True)
            use_compile: Enable torch.compile for better performance (requires PyTorch 2.3+)
            trust_remote_code: Whether to trust remote code when loading model (default: True)
            use_safetensors: Whether to use safetensors format (default: True)
            attn_implementation: Attention implementation to use (default: "eager")
            **model_kwargs: Additional keyword arguments to pass to AutoModel.from_pretrained()

        Returns:
            DeepSeekOCREncoder: Initialized encoder ready for inference

        Example:
            >>> from deepseek_ocr_encoder import DeepSeekOCREncoder
            >>> 
            >>> # One-line initialization
            >>> encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")
            >>> 
            >>> # Encode an image
            >>> tokens = encoder("document.pdf")  # Returns [1, N, 1024] tensor
            >>> 
            >>> # Custom device/dtype
            >>> encoder = DeepSeekOCREncoder.from_pretrained(
            ...     "deepseek-ai/DeepSeek-OCR",
            ...     device="cpu",
            ...     dtype=torch.float32
            ... )
            >>> 
            >>> # Load from local checkpoint
            >>> encoder = DeepSeekOCREncoder.from_pretrained("./my-finetuned-model")
        """
        # Auto-detect device if not specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Set default dtype based on device if not specified
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        # Load the full model from Hugging Face or local path
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_safetensors=use_safetensors,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            **model_kwargs,
        )
        
        # Set model to eval mode and move to device
        model = model.eval().to(device, dtype=dtype)

        # Create and return the encoder wrapper
        return cls(
            full_model=model,
            device=device,
            dtype=dtype,
            freeze=freeze,
            eager_to_device=eager_to_device,
            precompute_pos_for_1024=precompute_pos_for_1024,
            use_compile=use_compile,
        )

    @torch.inference_mode()
    def _forward_core(self, x_nchw_channels_last: torch.Tensor) -> torch.Tensor:
        """
        Core forward pass.

        Args:
            x_nchw_channels_last: Input tensor [B,3,1024,1024] in BF16, channels_last format

        Returns:
            Vision tokens [B, N, 1024] where N=Hs*Ws
        """
        sam_out = self.sam(x_nchw_channels_last)  # [B, 1024, Hs, Ws]
        B, C, Hs, Ws = sam_out.shape
        if C != self.embed_dim:
            raise RuntimeError(f"Expected {self.embed_dim} channels before CLIP, got {C}")

        tokens = sam_out.flatten(2).transpose(1, 2).contiguous()  # [B, N, 1024], N=Hs*Ws

        if Hs == 16 and Ws == 16 and self._pos_fixed_16 is not None:
            pos = self._pos_fixed_16  # [1,256,1024]
        else:
            key = (Hs, Ws)
            pos = self._pos_cache.get(key)
            if pos is None:
                table = self.clip_pos_table
                grid_size = int(math.isqrt(table.size(0) - 1))
                base_grid = table[1: 1 + grid_size * grid_size].view(grid_size, grid_size, self.embed_dim)
                base_grid = base_grid.permute(2, 0, 1).unsqueeze(0)  # [1,1024,G,G]
                if (Hs, Ws) != (grid_size, grid_size):
                    base_grid = F.interpolate(base_grid, size=(Hs, Ws), mode="bicubic", align_corners=False)
                pos = base_grid.flatten(2).transpose(1, 2).contiguous()  # [1,N,1024]
                self._pos_cache[key] = pos

        tokens_plus = tokens + pos  # broadcast over batch
        x_tok = self.clip_pre(tokens_plus)
        x_tok = self.clip_tr(x_tok)
        return tokens + x_tok  # [B,N,1024]

    @torch.inference_mode()
    def encode(self, image: Union[Image.Image, str, "os.PathLike"]) -> torch.Tensor:
        """
        Encode an image into vision tokens.

        Args:
            image: PIL Image or path to an RGB image file

        Returns:
            Vision tokens tensor of shape [1, N, 1024] where N=256 for 1024x1024 input
        """
        if isinstance(image, (str, bytes)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # CPU preproc → pinned → non_blocking H2D
        x_cpu = self._preproc_1024(img).unsqueeze(0).pin_memory()  # [1,3,1024,1024] on CPU
        x = x_cpu.to(self.device, dtype=self.dtype, non_blocking=True)  # -> GPU BF16
        x = x.to(memory_format=torch.channels_last)  # NHWC memory layout

        # Fast path: CUDA Graph replay if captured
        if self._graph is not None:
            self._static_in.copy_(x)  # copy into static buffer
            self._graph.replay()
            return self._static_out

        # Compiled path if enabled; otherwise plain
        if self._compiled is not None:
            return self._compiled(x)
        else:
            return self._forward_core(x)

    @torch.inference_mode()
    def capture_cudagraph(self, batch_size: int = 1, H: int = 1024, W: int = 1024):
        """
        Capture a CUDA graph for optimized steady-state inference.

        Call this once after initialization to enable CUDA graph acceleration.
        Subsequent calls to .encode() will reuse the captured graph.

        Args:
            batch_size: Batch size for the graph (default: 1)
            H: Input height (default: 1024)
            W: Input width (default: 1024)

        Raises:
            RuntimeError: If device is not CUDA
        """
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graphs require CUDA device.")
        if self._graph is not None:
            return  # already captured

        # Static input/output buffers (must not resize later)
        static_in = torch.empty(
            batch_size, 3, H, W, device=self.device, dtype=self.dtype
        ).to(memory_format=torch.channels_last)
        
        # Warm-up to materialize kernels & memory pools
        for _ in range(3):
            _ = self._forward_core(static_in)

        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            static_out = self._forward_core(static_in)
        self._graph = g
        self._static_in = static_in
        self._static_out = static_out

    def __call__(self, image: Union[Image.Image, str, "os.PathLike"]) -> torch.Tensor:
        """
        Convenience method to encode an image.

        Args:
            image: PIL Image or path to an RGB image file

        Returns:
            Vision tokens tensor of shape [1, N, 1024]
        """
        return self.encode(image)
