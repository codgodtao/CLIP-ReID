"""
MobileCLIP2-based ReID model factory.

This module adapts Apple's MobileCLIP2 for person/vehicle re-identification,
replacing the original OpenAI CLIP backbone. Key adaptations:

1. **FastViT reparameterization**: MobileCLIP2 uses FastViT with structural
   reparameterization — multi-branched during training, single-branched at
   inference. We keep the multi-branched structure during both Stage 1 (prompt
   learning) and Stage 2 (vision encoder fine-tuning) so BatchNorm statistics
   can update. Reparameterization is only applied at inference time.

2. **Intermediate feature extraction**: The FastViT vision encoder is modified
   to return three feature maps (second-to-last block, last block, projected)
   to support the triplet + ID + I2T loss used in CLIP-ReID.

3. **Text encoder**: A standard 12-layer transformer (matching MobileCLIP2's
   OpenCLIP-format text encoder) replaces the original 4-layer mobileclip v1
   text encoder. Supports both causal and non-causal masking.

4. **PromptLearner**: Per-class learnable context tokens, same as CLIP-ReID.
"""
import copy
import logging
import math
from collections import OrderedDict
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from timm.models.layers import trunc_normal_

# Register FastViT models (mci0, mci1, mci2, vit_b16) with timm
import mobileclip.models  # noqa: F401
from mobileclip.modules.common.mobileone import reparameterize_model
from mobileclip.modules.image.image_projection import GlobalPool2D, GlobalPool

# Register S3/S4 models if available
try:
    import mobileclip2.mobileclip2  # noqa: F401
except ImportError:
    pass

import open_clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
logger = logging.getLogger("transreid.model")


# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------
# Each variant maps to a timm model name and specifies the dimensions needed
# to build the ReID head (BNNeck + classifiers).
#
# For FastViT-based variants (S0, S2):
#   - vision_dim = last stage embed_dim (before conv_exp)
#   - After conv_exp: vision_dim * cls_ratio (default 2.0)
#   - GlobalPool2D projects from (vision_dim * cls_ratio) to embed_dim
#
# For ViT-based variants (B):
#   - vision_dim = transformer embed_dim
#   - SimpleImageProjectionHead projects from vision_dim to embed_dim
# ---------------------------------------------------------------------------

MOBILECLIP2_VARIANTS: Dict[str, dict] = {
    "MobileCLIP2-S0": {
        "timm_name": "mci0",
        "image_size": 256,
        "embed_dim": 512,
        "vision_dim": 512,  # last stage embed_dim for mci0
        "cls_ratio": 2.0,
        "text_width": 512,
        "text_heads": 8,
        "text_layers": 12,
        "context_length": 77,
        "vocab_size": 49408,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "no_causal_mask": True,
        "is_fastvit": True,
    },
    "MobileCLIP2-S2": {
        "timm_name": "mci2",
        "image_size": 256,
        "embed_dim": 512,
        "vision_dim": 640,  # last stage embed_dim for mci2
        "cls_ratio": 2.0,
        "text_width": 512,
        "text_heads": 8,
        "text_layers": 12,
        "context_length": 77,
        "vocab_size": 49408,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "no_causal_mask": True,
        "is_fastvit": True,
    },
    "MobileCLIP2-B": {
        "timm_name": "vit_b16",  # registered by mobileclip
        "image_size": 224,
        "embed_dim": 512,
        "vision_dim": 768,
        "cls_ratio": 1.0,  # ViT uses SimpleImageProjectionHead, no conv_exp
        "text_width": 512,
        "text_heads": 8,
        "text_layers": 12,
        "context_length": 77,
        "vocab_size": 49408,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "no_causal_mask": False,
        "is_fastvit": False,
    },
}


# ---------------------------------------------------------------------------
# Text encoder (OpenCLIP-format 12-layer transformer)
# ---------------------------------------------------------------------------
class ResidualAttentionBlock(nn.Module):
    """Standard transformer block matching OpenCLIP's text encoder structure.

    Uses ``OrderedDict`` naming (``c_fc``, ``gelu``, ``c_proj``) for the MLP
    so that OpenCLIP-format MobileCLIP2 checkpoints load correctly.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class TextTransformer(nn.Module):
    """12-layer text transformer matching MobileCLIP2's OpenCLIP text encoder.

    Supports both causal masking (original CLIP style) and bidirectional
    attention (MobileCLIP2-S0/S2 style via ``no_causal_mask``).
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int,
        vocab_size: int,
        embed_dim: int,
        no_causal_mask: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.width = width
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, width)
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        trunc_normal_(self.positional_embedding, std=0.02)

        # Attention mask (causal or None)
        if not no_causal_mask:
            mask = torch.empty(context_length, context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            self.register_buffer("attn_mask", mask, persistent=False)
        else:
            self.attn_mask = None

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, self.attn_mask) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        trunc_normal_(self.text_projection, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=0.02)

    def forward_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Token embedding only (positional embedding added in encode_text_from_embeddings)."""
        token_emb = self.token_embedding(text_tokens)
        return token_emb

    def encode_text_from_embeddings(self, embeddings: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """Encode text from pre-computed token embeddings (for prompt learning).

        Adds positional embedding to the input embeddings before running the
        transformer, matching the original CLIP-ReID TextEncoder behavior.

        Args:
            embeddings: [batch_size, context_length, width] — token embeddings
                (WITHOUT positional embedding; it is added here).
            tokenized_prompts: [batch_size, context_length] — tokenized prompts
                (used to find the EOT token position).

        Returns:
            [batch_size, embed_dim] — text features.
        """
        # Add positional embedding (same as original CLIP-ReID TextEncoder)
        x = embeddings + self.positional_embedding.to(embeddings.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # Take features from the EOT token
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.forward_embedding(text_tokens)
        return self.encode_text_from_embeddings(embeddings, text_tokens)


# ---------------------------------------------------------------------------
# Vision encoder wrappers
# ---------------------------------------------------------------------------
class FastViTVisionEncoder(nn.Module):
    """Wraps a FastViT model for ReID, extracting intermediate features.

    The FastViT model's forward pass is modified to return three features:
    - ``feat_last``: GAP of second-to-last attention block output (triplet loss)
    - ``feat``: GAP of last attention block output (ID loss + triplet loss)
    - ``feat_proj``: Projected feature after GlobalPool2D (I2T matching)

    The multi-branched structure is preserved during training (no
    reparameterization). Call :meth:`reparameterize` for inference.
    """

    def __init__(self, timm_model_name: str, embed_dim: int, vision_dim: int, cls_ratio: float = 2.0):
        super().__init__()
        # Create the FastViT model (training mode, multi-branched)
        self.model = create_model(
            timm_model_name,
            projection_dim=embed_dim,
            num_classes=0,  # remove classification head
        )
        # Replace the head with GlobalPool2D (same as MCi wrapper)
        conv_exp_dim = int(vision_dim * cls_ratio)
        self.model.head = GlobalPool2D(in_dim=conv_exp_dim, out_dim=embed_dim)
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim

        # Identify the last stage (last nn.Sequential in network)
        self._last_stage_idx = self._find_last_stage()

    def _find_last_stage(self) -> int:
        """Find the index of the last stage (nn.Sequential of blocks) in network."""
        network = self.model.network
        last_seq_idx = None
        for idx in range(len(network) - 1, -1, -1):
            if isinstance(network[idx], nn.Sequential):
                last_seq_idx = idx
                break
        if last_seq_idx is None:
            raise RuntimeError("Could not find last stage in FastViT network")
        return last_seq_idx

    def forward(self, x: torch.Tensor, cv_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning intermediate features for ReID.

        Args:
            x: Input images [B, 3, H, W].
            cv_emb: Optional side-information embedding [B, vision_dim] for SIE.
                Added to spatial features before global pooling.

        Returns:
            (feat_last, feat, feat_proj) where:
            - feat_last: [B, vision_dim] — second-to-last block features
            - feat: [B, vision_dim] — last block features
            - feat_proj: [B, embed_dim] — projected features
        """
        # Stem
        x = self.model.forward_embeddings(x)

        # Run through network up to (but not including) the last stage
        network = self.model.network
        for idx in range(self._last_stage_idx):
            x = network[idx](x)

        # Split the last stage to extract intermediate features
        last_stage = network[self._last_stage_idx]
        num_blocks = len(last_stage)

        # Run through all blocks except the last one
        for i in range(num_blocks - 1):
            x = last_stage[i](x)
        feat_last_spatial = x  # [B, C, H, W]

        # Run through the last block
        x = last_stage[-1](x)
        feat_spatial = x  # [B, C, H, W]

        # Apply conv_exp
        x = self.model.conv_exp(x)  # [B, C*cls_ratio, H, W]

        # Apply GlobalPool2D (head) to get projected features
        feat_proj = self.model.head(x)  # [B, embed_dim]

        # Global average pool the spatial features
        feat_last = feat_last_spatial.mean(dim=[2, 3])  # [B, vision_dim]
        feat = feat_spatial.mean(dim=[2, 3])  # [B, vision_dim]

        # Add SIE (Side Information Embedding) if provided
        if cv_emb is not None:
            # Broadcast cv_emb [B, C] to [B, C, 1, 1] and add to spatial features
            feat_last = feat_last + cv_emb
            feat = feat + cv_emb

        return feat_last, feat, feat_proj

    def reparameterize(self):
        """Fold multi-branched structure into single-branch for inference.

        Must be called after ``model.eval()`` (BatchNorm running stats are used).
        """
        self.model = reparameterize_model(self.model)


class ViTVisionEncoder(nn.Module):
    """Wraps a ViT-based model (MobileCLIP2-B) for ReID.

    Uses the CLS token as the global feature, similar to the original CLIP-ReID.
    """

    def __init__(self, timm_model_name: str, embed_dim: int, vision_dim: int):
        super().__init__()
        from mobileclip.models.vit import VisionTransformer

        cfg = {
            "norm_layer": "layer_norm_fp32",
            "act_layer": "gelu",
            "embed_dim": vision_dim,
            "n_transformer_layers": 12,
            "n_attn_heads": 12,
        }
        self.model = VisionTransformer(cfg=cfg, projection_dim=embed_dim)
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, cv_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract features at different depths
        # Run through patch embedding
        patch_emb, (n_h, n_w) = self.model.extract_patch_embeddings(x)

        # Run through transformer layers, capturing intermediate outputs
        x = patch_emb
        feat_last = None
        for i, layer in enumerate(self.model.transformer):
            x = layer(x)
            if i == len(self.model.transformer) - 2:
                # Second-to-last layer output
                x_last = self.model.post_transformer_norm(x) if hasattr(self.model, 'post_transformer_norm') else x
                feat_last = x_last[:, 0] if self.model.cls_token is not None else x_last.mean(dim=1)

        # Final layer output
        x = self.model.post_transformer_norm(x)
        feat = x[:, 0] if self.model.cls_token is not None else x.mean(dim=1)

        # Projected features
        feat_proj = self.model.classifier(feat)

        # Add SIE if provided
        if cv_emb is not None:
            feat_last = feat_last + cv_emb
            feat = feat + cv_emb

        return feat_last, feat, feat_proj

    def reparameterize(self):
        """No reparameterization needed for ViT models."""
        pass


# ---------------------------------------------------------------------------
# Prompt learner (same as CLIP-ReID, adapted for new text encoder)
# ---------------------------------------------------------------------------
class PromptLearner(nn.Module):
    """Per-class learnable context tokens for prompt learning (Stage 1).

    Same structure as the original CLIP-ReID PromptLearner, but uses the
    MobileCLIP2 text encoder's token embedding.
    """

    def __init__(self, num_class: int, dataset_name: str, text_encoder: TextTransformer):
        super().__init__()
        if dataset_name in ("VehicleID", "veri"):
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = text_encoder.width
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        # Tokenize the prompt template with SOT/EOT tokens (same as clip.tokenize)
        # SimpleTokenizer.encode() does NOT add SOT/EOT, so we add them manually.
        # SOT = 49406, EOT = 49407 (last two tokens in CLIP vocab)
        sot_token = 49406
        eot_token = 49407
        tokenized_prompts = [sot_token] + _tokenizer.encode(ctx_init) + [eot_token]
        # Pad/truncate to context_length
        context_length = text_encoder.context_length
        if len(tokenized_prompts) > context_length:
            tokenized_prompts = tokenized_prompts[:context_length]
        else:
            tokenized_prompts = tokenized_prompts + [0] * (context_length - len(tokenized_prompts))
        tokenized_prompts = torch.tensor(tokenized_prompts, dtype=torch.long).unsqueeze(0).cuda()

        with torch.no_grad():
            # Only token embedding here; positional embedding is added in
            # encode_text_from_embeddings (same as original CLIP-ReID).
            embedding = text_encoder.token_embedding(tokenized_prompts)  # [1, CL, dim]

        self.tokenized_prompts = tokenized_prompts
        self.n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, self.n_cls_ctx, ctx_dim)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # Store prefix and suffix (non-learnable, token embedding only)
        # prefix: SOT + "a photo of a " (n_ctx + 1 tokens)
        # suffix: " person/vehicle." + EOT + padding
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + self.n_cls_ctx:, :])
        self.num_class = num_class

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        cls_ctx = self.cls_ctx[label]  # [B, n_cls_ctx, dim]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)
        return prompts


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------
def load_mobileclip2_weights(model, variant_name: str, pretrained_path: str = "", hf_download: bool = True):
    """Load MobileCLIP2 pretrained weights into the ReID model.

    Handles key mapping from OpenCLIP checkpoint format to our model structure.

    Args:
        model: The build_transformer model instance.
        variant_name: e.g. "MobileCLIP2-S0".
        pretrained_path: Direct path to checkpoint file. If empty, downloads from HuggingFace.
        hf_download: If True and pretrained_path is empty, download from HuggingFace.
    """
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
    elif hf_download:
        try:
            from huggingface_hub import hf_hub_download
            hf_name = f"apple/{variant_name}"
            logger.info(f"Downloading {variant_name} weights from HuggingFace...")
            pretrained_path = hf_hub_download(repo_id=hf_name, filename="open_clip_pytorch_model.pt")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Could not download from HuggingFace: {e}")
            logger.warning("Model will use random initialization. Please provide pretrained weights manually.")
            return
    else:
        logger.warning("No pretrained weights provided. Using random initialization.")
        return

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Map OpenCLIP keys to our model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # Vision encoder keys: visual.trunk.* -> image_encoder.model.*
        if key.startswith("visual.trunk."):
            new_key = key.replace("visual.trunk.", "image_encoder.model.", 1)
        # Vision projection: visual.head.* -> image_encoder.model.head.*
        elif key.startswith("visual.head."):
            new_key = key.replace("visual.head.", "image_encoder.model.head.", 1)
        # Text encoder keys: text.* -> text_encoder.*
        elif key.startswith("text."):
            new_key = key.replace("text.", "text_encoder.", 1)
        # Logit scale
        elif key == "logit_scale":
            new_key = "logit_scale"

        new_state_dict[new_key] = value

    # Load with strict=False to handle any remaining mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading pretrained: {len(missing)} keys")
        # Show first 10 missing keys
        for k in missing[:10]:
            logger.warning(f"  Missing: {k}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading pretrained: {len(unexpected)} keys")
        for k in unexpected[:10]:
            logger.warning(f"  Unexpected: {k}")

    logger.info(f"Loaded MobileCLIP2 pretrained weights from {pretrained_path or 'HuggingFace'}")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    """MobileCLIP2-based ReID model with two-stage training support.

    Stage 1: Only ``prompt_learner`` is trained (vision + text frozen).
    Stage 2: Vision encoder + BNNeck + classifiers are trained (text + prompt frozen).
    """

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num

        # Get variant config
        if self.model_name not in MOBILECLIP2_VARIANTS:
            raise ValueError(
                f"Unknown model name: {self.model_name}. "
                f"Supported: {list(MOBILECLIP2_VARIANTS.keys())}"
            )
        variant_cfg = MOBILECLIP2_VARIANTS[self.model_name]
        self.variant_cfg = variant_cfg

        # Dimensions
        self.in_planes = variant_cfg["vision_dim"]  # raw feature dim
        self.in_planes_proj = variant_cfg["embed_dim"]  # projected feature dim
        self.sie_coe = cfg.MODEL.SIE_COE

        # BNNeck + classifiers
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # Build vision encoder
        if variant_cfg["is_fastvit"]:
            self.image_encoder = FastViTVisionEncoder(
                timm_model_name=variant_cfg["timm_name"],
                embed_dim=variant_cfg["embed_dim"],
                vision_dim=variant_cfg["vision_dim"],
                cls_ratio=variant_cfg["cls_ratio"],
            )
        else:
            self.image_encoder = ViTVisionEncoder(
                timm_model_name=variant_cfg["timm_name"],
                embed_dim=variant_cfg["embed_dim"],
                vision_dim=variant_cfg["vision_dim"],
            )

        # SIE (Side Information Embedding)
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        else:
            self.cv_embed = None

        # Build text encoder
        self.text_encoder = TextTransformer(
            width=variant_cfg["text_width"],
            layers=variant_cfg["text_layers"],
            heads=variant_cfg["text_heads"],
            context_length=variant_cfg["context_length"],
            vocab_size=variant_cfg["vocab_size"],
            embed_dim=variant_cfg["embed_dim"],
            no_causal_mask=variant_cfg["no_causal_mask"],
        )

        # Logit scale (from CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

        # Prompt learner
        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, self.text_encoder)

        # Load pretrained weights
        pretrained_path = getattr(cfg.MODEL, "PRETRAIN_PATH", "")
        if cfg.MODEL.PRETRAIN_CHOICE == "imagenet" or cfg.MODEL.PRETRAIN_CHOICE == "self":
            load_mobileclip2_weights(self, self.model_name, pretrained_path)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        get_image: bool = False,
        get_text: bool = False,
        cam_label: Optional[torch.Tensor] = None,
        view_label: Optional[torch.Tensor] = None,
    ):
        # --- Text feature extraction (Stage 1 prompt learning) ---
        if get_text:
            prompts = self.prompt_learner(label)
            # prompts are token embeddings (no pos_emb);
            # pos_emb is added inside encode_text_from_embeddings
            text_features = self.text_encoder.encode_text_from_embeddings(
                prompts, self.prompt_learner.tokenized_prompts
            )
            return text_features

        # --- Image feature extraction (Stage 1 cache) ---
        if get_image:
            _, _, feat_proj = self.image_encoder(x)
            return feat_proj

        # --- Full forward (Stage 2 training / inference) ---
        # SIE
        cv_emb = None
        if self.cv_embed is not None:
            if cam_label is not None and view_label is not None:
                cv_emb = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_emb = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_emb = self.sie_coe * self.cv_embed[view_label]

        feat_last, feat, feat_proj = self.image_encoder(x, cv_emb)

        # BNNeck
        feat_bn = self.bottleneck(feat)
        feat_proj_bn = self.bottleneck_proj(feat_proj)

        if self.training:
            cls_score = self.classifier(feat_bn)
            cls_score_proj = self.classifier_proj(feat_proj_bn)
            return [cls_score, cls_score_proj], [feat_last, feat, feat_proj], feat_proj
        else:
            if self.neck_feat == "after":
                return torch.cat([feat_bn, feat_proj_bn], dim=1)
            else:
                return torch.cat([feat, feat_proj], dim=1)

    def reparameterize(self):
        """Fold FastViT multi-branched structure for inference speed.

        Call after ``model.eval()``. Cannot be undone — the model cannot
        be trained after reparameterization.
        """
        logger.info("Reparameterizing FastViT for inference...")
        self.image_encoder.reparameterize()
        logger.info("Reparameterization complete.")

    def load_param(self, trained_path: str):
        """Load trained ReID checkpoint."""
        param_dict = torch.load(trained_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            key = i.replace("module.", "")
            if key in self.state_dict():
                self.state_dict()[key].copy_(param_dict[i])
            else:
                logger.warning(f"Key not found in model: {key}")
        logger.info(f"Loading pretrained model from {trained_path}")

    def load_param_finetune(self, model_path: str):
        """Load checkpoint for fine-tuning (partial match)."""
        param_dict = torch.load(model_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            key = i.replace("module.", "")
            if key in self.state_dict():
                self.state_dict()[key].copy_(param_dict[i])
        logger.info(f"Loading pretrained model for finetuning from {model_path}")


def make_model(cfg, num_class, camera_num, view_num):
    """Factory function to create a MobileCLIP2-based ReID model."""
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model
