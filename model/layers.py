"""
Shared Components for D3-DNA Discrete Diffusion Models

This module contains core components that are shared between transformer and 
convolutional architectures. These components are dataset-agnostic and should 
never need modification when adding new datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    modulate_fused,
)


def modulate(x, shift, scale):
    """Modulate input with shift and scale parameters."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def residual_linear(x, W, x_skip, residual_scale):
    """Compute residual linear transformation: x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation with autocast handling."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
            
        Returns:
            (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations with classifier-free guidance support.
    """
    
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply label dropout for classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, 
                force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# class EmbeddingLayer(nn.Module):
#     """
#     Generic embedding layer that combines vocabulary and signal embeddings.
    
#     This is dataset-agnostic - signal_dim is passed as a parameter.
#     """
    
#     def __init__(self, dim: int, vocab_dim: int, signal_dim: int):
#         super().__init__()
#         self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
#         self.signal_embedding = nn.Linear(signal_dim, dim)
#         torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Sequence tokens (batch_size, seq_length)
#             y: Signal embeddings (batch_size, signal_dim) or (batch_size, seq_length, signal_dim)
            
#         Returns:
#             Combined embeddings (batch_size, seq_length, dim)
#         """
#         vocab_embed = self.embedding[x]
#         signal_embed = self.signal_embedding(y.to(torch.float32))
        
#         # Handle broadcasting for different signal dimensions
#         if signal_embed.dim() == 2 and vocab_embed.dim() == 3:
#             signal_embed = signal_embed[:, None, :]  # Broadcast to sequence length
#         elif signal_embed.dim() == 3 and vocab_embed.dim() == 3:
#             pass  # Already correct dimensions
#         else:
#             raise ValueError(f"Incompatible dimensions: vocab_embed {vocab_embed.shape}, signal_embed {signal_embed.shape}")
        
#         return torch.add(vocab_embed, signal_embed)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer for convolutional architectures."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[...]


def get_bias_dropout_scale():
    """Get the appropriate bias dropout scale function based on training state."""
    def _get_bias_dropout_scale(training: bool):
        return (
            bias_dropout_add_scale_fused_train
            if training
            else bias_dropout_add_scale_fused_inference
        )
    return _get_bias_dropout_scale