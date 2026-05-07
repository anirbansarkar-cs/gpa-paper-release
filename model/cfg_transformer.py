"""
CFG Transformer Architecture for D3-DNA Discrete Diffusion

This module implements Classifier-Free Guidance (CFG) for the SEDD model on DNA.
It replaces the linear signal_embedding with Random Fourier Features + Cross-Attention,
and adds CFG support through condition dropout during training and dual forward passes
during sampling.

Key classes:
- ScalarConditioner: Fourier Feature encoder for activity values
- CrossAttention: Activity conditioning sublayer in each block
- CFGDDiTBlock: Modified Transformer block (Self-Attn -> Cross-Attn -> FFN)
- CFGEmbeddingLayer: Token-only embedding (no signal mixing at input)
- CFGTransformerModel: Main model class with CFG support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import OmegaConf, DictConfig
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func

from . import rotary
from .layers import (
    LayerNorm, TimestepEmbedder,
    modulate_fused, get_bias_dropout_scale
)
from .transformer import DDitFinalLayer


class ScalarConditioner(nn.Module):
    """
    Encodes scalar activity values into conditioning vectors using
    Random Fourier Features + MLP.

    Replaces the linear signal_embedding with a richer representation.
    Supports z-score normalization and null embedding for unconditional case.

    Args:
        signal_dim: Number of activity dimensions (1 for LentiMPRA).
        cond_dim: Output conditioning dimension.
        embedding_size: Size of Fourier feature embedding per signal dimension.
    """

    def __init__(self, signal_dim: int, cond_dim: int, embedding_size: int = 256):
        super().__init__()
        self.signal_dim = signal_dim
        self.cond_dim = cond_dim
        self.embedding_size = embedding_size

        # Random Fourier Feature frequencies (fixed, not learned)
        self.register_buffer('freqs', torch.randn(signal_dim, embedding_size // 2))

        # MLP: Fourier features -> conditioning vector
        fourier_out_dim = signal_dim * embedding_size  # sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(fourier_out_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Null embedding for unconditional case (CFG dropout)
        self.null_embedding = nn.Parameter(torch.zeros(cond_dim))

        # Z-score normalization buffers
        self.register_buffer('activity_mean', torch.zeros(signal_dim))
        self.register_buffer('activity_std', torch.ones(signal_dim))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set z-score normalization statistics from training data."""
        self.activity_mean.copy_(mean.reshape(self.signal_dim))
        self.activity_std.copy_(std.reshape(self.signal_dim))

    def forward(self, activity: torch.Tensor,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            activity: (batch, signal_dim) raw activity values.
            drop_mask: (batch,) bool tensor. True -> return null_embedding.

        Returns:
            (batch, cond_dim) conditioning vector.
        """
        # Z-score normalize
        activity = (activity.float() - self.activity_mean) / (self.activity_std + 1e-8)

        # Random Fourier Features: (batch, signal_dim) x (signal_dim, E//2) -> (batch, signal_dim, E//2)
        proj = activity.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        # -> (batch, signal_dim * E) via sin/cos concat then flatten
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (batch, signal_dim, E)
        fourier = fourier.view(activity.shape[0], -1)  # (batch, signal_dim * E)

        # MLP
        cond = self.mlp(fourier)  # (batch, cond_dim)

        # Apply dropout mask
        if drop_mask is not None:
            # drop_mask: (batch,) bool
            null = self.null_embedding.unsqueeze(0).expand(cond.shape[0], -1)
            cond = torch.where(drop_mask.unsqueeze(-1), null, cond)

        return cond


class CrossAttention(nn.Module):
    """
    Cross-attention sublayer for activity conditioning.

    Q comes from sequence hidden states, K/V from conditioning vector.
    The conditioning is a single memory token: (batch, 1, cond_dim).
    Uses standard PyTorch attention since KV length=1.

    Has its own LayerNorm + adaLN modulation gated by time embedding c_time.
    adaLN weights are zero-initialized so cross-attention starts as identity.

    Args:
        dim: Hidden dimension (768).
        n_heads: Number of attention heads (12).
        cond_dim: Conditioning dimension (128).
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        # Layer norm for sequence input
        self.norm = LayerNorm(dim)

        # Q projection from sequence hidden states
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # K, V projections from conditioning (cond_dim -> dim)
        self.k_proj = nn.Linear(cond_dim, dim, bias=False)
        self.v_proj = nn.Linear(cond_dim, dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # adaLN modulation gated by time: shift, scale, gate (3 * dim)
        self.adaLN_modulation = nn.Linear(cond_dim, 3 * dim, bias=True)
        # Zero-initialize so cross-attention starts as identity
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c_activity: torch.Tensor,
                c_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) sequence hidden states.
            c_activity: (batch, cond_dim) activity conditioning.
            c_time: (batch, cond_dim) time conditioning.

        Returns:
            (batch, seq_len, dim) residual-added output.
        """
        batch_size, seq_len, dim = x.shape

        # adaLN modulation from time
        shift, scale, gate = self.adaLN_modulation(c_time)[:, None].chunk(3, dim=2)

        # Normalize and modulate
        x_norm = modulate_fused(self.norm(x), shift, scale)

        # Q from sequence: (batch, seq_len, dim) -> (batch, n_heads, seq_len, head_dim)
        q = self.q_proj(x_norm)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)

        # K, V from conditioning: (batch, cond_dim) -> (batch, 1, dim) -> (batch, n_heads, 1, head_dim)
        c_act = c_activity.unsqueeze(1)  # (batch, 1, cond_dim)
        k = self.k_proj(c_act)
        v = self.v_proj(c_act)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        # Standard attention (KV length=1, so no need for Flash Attention)
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale_factor  # (batch, n_heads, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)

        # Reshape back
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        attn_out = self.out_proj(attn_out)

        # Gated residual connection
        x = x + gate * attn_out

        return x


class CFGDDiTBlock(nn.Module):
    """
    Modified Transformer block: Self-Attn -> Cross-Attn -> FFN.

    Self-attention portion is identical to DDiTBlock (Flash Attn + RoPE + adaLN from c_time).
    Cross-attention calls CrossAttention(x, c_activity, c_time) between self-attn and FFN.
    FFN portion is identical to DDiTBlock.

    Args:
        dim: Hidden dimension (768).
        n_heads: Number of attention heads (12).
        cond_dim: Conditioning dimension (128).
        mlp_ratio: FFN expansion ratio (4).
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        # === Self-Attention (identical to DDiTBlock) ===
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # === Cross-Attention ===
        self.cross_attn = CrossAttention(dim, n_heads, cond_dim, dropout)

        # === Feed-forward (identical to DDiTBlock) ===
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        # adaLN modulation for self-attention + FFN (6 * dim: shift/scale/gate for SA and FFN)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return get_bias_dropout_scale()(self.training)

    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
                c_time: torch.Tensor, c_activity: torch.Tensor,
                seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) input tensor.
            rotary_cos_sin: Rotary position encoding (cos, sin).
            c_time: (batch, cond_dim) time conditioning.
            c_activity: (batch, cond_dim) activity conditioning.
            seqlens: Optional sequence lengths for variable-length attention.

        Returns:
            (batch, seq_len, dim) output tensor.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # adaLN modulation from time (for self-attention and FFN)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c_time)[:, None].chunk(6, dim=2)

        # === 1. Self-Attention + adaLN(c_time) ===
        x_skip = x
        x_sa = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x_sa)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        # Apply rotary position embedding
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        qkv = qkv.contiguous()

        # Flash attention
        if seqlens is None:
            x_sa = flash_attn_qkvpacked_func(qkv, dropout_p=0., causal=False)
            x_sa = rearrange(x_sa, 'b s h d -> b s (h d)')
        else:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            cu_seqlens = seqlens.cumsum(-1)
            x_sa = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0., causal=False)
            x_sa = rearrange(x_sa, '(b s) h d -> b s (h d)', b=batch_size)

        # Gated residual for self-attention
        x = bias_dropout_scale_fn(self.attn_out(x_sa), None, gate_msa, x_skip, self.dropout)

        # === 2. Cross-Attention (c_activity, gated by c_time) ===
        x = self.cross_attn(x, c_activity, c_time)

        # === 3. FFN + adaLN(c_time) ===
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout
        )

        return x


class CFGEmbeddingLayer(nn.Module):
    """
    Token-only embedding layer (no signal mixing at input).

    Activity conditioning is handled entirely through cross-attention
    in the transformer blocks, not at the embedding level.

    Args:
        dim: Embedding dimension (768).
        vocab_dim: Vocabulary size (4 + 1 for absorbing state).
    """

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices.

        Returns:
            (batch, seq_len, dim) vocabulary embeddings.
        """
        return self.embedding[x]


class CFGTransformerModel(nn.Module):
    """
    CFG-enabled Transformer SEDD Model for DNA.

    Combines Cross-Attention for activity conditioning and Classifier-Free Guidance
    for steering during sampling. Activity is encoded via Random Fourier Features
    and injected through cross-attention in every transformer block.

    During training, 15% of samples randomly get null conditioning (CFG dropout).
    During sampling, dual forward passes (conditional + unconditional) enable
    logit-level CFG.

    Forward signature is compatible with existing get_model_fn/get_score_fn.

    Args:
        config: Configuration object with model, dataset, and graph sections.
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config

        # Extract parameters
        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        signal_dim = config.dataset.signal_dim

        # CFG parameters
        self.cond_dropout_prob = getattr(config.model, 'cond_dropout_prob', 0.15)
        fourier_embedding_size = getattr(config.model, 'fourier_embedding_size', 256)

        # Token-only embedding (no signal at input)
        self.vocab_embed = CFGEmbeddingLayer(
            dim=config.model.hidden_size,
            vocab_dim=vocab_size,
        )

        # Time conditioning
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)

        # Activity conditioning via Fourier Features
        self.activity_conditioner = ScalarConditioner(
            signal_dim=signal_dim,
            cond_dim=config.model.cond_dim,
            embedding_size=fourier_embedding_size,
        )

        # Rotary position embeddings
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            CFGDDiTBlock(
                dim=config.model.hidden_size,
                n_heads=config.model.n_heads,
                cond_dim=config.model.cond_dim,
                dropout=config.model.dropout,
            )
            for _ in range(config.model.n_blocks)
        ])

        # Output layer (same as original, conditioned on c_time only)
        self.output_layer = DDitFinalLayer(
            hidden_size=config.model.hidden_size,
            out_channels=vocab_size,
            cond_dim=config.model.cond_dim,
        )

        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)

    def forward(self, indices: torch.Tensor, labels: Optional[torch.Tensor] = None,
                train: bool = True, sigma: Optional[torch.Tensor] = None,
                force_drop_ids: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with CFG support.

        Args:
            indices: (batch, seq_len) token indices.
            labels: (batch, signal_dim) activity values, or None for unconditional.
            train: Training mode flag.
            sigma: (batch,) noise level.
            force_drop_ids: (batch,) bool tensor. When True, forces unconditional pass.
                           Used for CFG sampling to get the unconditional output.
            layer_idx: Index of the layer to return the representation of.

        Returns:
            (logits, rep) where logits is (batch, seq_len, vocab_size) and
            rep is the representation at layer_idx (or None).
        """
        batch_size = indices.shape[0]

        # Token embedding (no signal mixing)
        x = self.vocab_embed(indices)

        # Time conditioning
        c_time = F.silu(self.sigma_map(sigma))

        # Activity conditioning with CFG dropout
        if labels is not None:
            # Determine dropout mask
            if force_drop_ids is not None:
                drop_mask = force_drop_ids.bool()
            elif train and self.cond_dropout_prob > 0:
                drop_mask = torch.rand(batch_size, device=indices.device) < self.cond_dropout_prob
            else:
                drop_mask = None

            c_activity = self.activity_conditioner(labels, drop_mask=drop_mask)
        else:
            # Fully unconditional: use null embedding for all
            c_activity = self.activity_conditioner.null_embedding.unsqueeze(0).expand(batch_size, -1)

        # Rotary position encoding
        rotary_cos_sin = self.rotary_emb(x)

        # Forward through transformer blocks
        rep = None
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i, block in enumerate(self.blocks):
                x = block(x, rotary_cos_sin, c_time, c_activity, seqlens=None)
                if layer_idx is not None and i == layer_idx:
                    rep = x

            x = self.output_layer(x, c_time)

        # Mask out the input tokens (standard diffusion technique)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x, rep
