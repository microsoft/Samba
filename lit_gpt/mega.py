# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from .s6 import S6
from lit_gpt.rmsnorm import FusedRMSNorm

class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, embed_dim))
        self.beta = nn.Parameter(torch.Tensor(1, embed_dim))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.alpha, mean=0.0, std=std)
        nn.init.normal_(self.beta, mean=0.0, std=std)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x):
        n, d = x.size()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.sine is None or n > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[:n]
        cos = self.cosine[:n]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

    def forward(self, seq_len):
        a = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        b = self.rotary(self.beta.expand(seq_len, self.embed_dim))
        t = torch.einsum('mk,nk->mn', a, b)
        return t

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}'.format(self.embed_dim, self.max_positions)
    

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class S6GatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim = 128,
        ndim = 16,
        chunk_size=2048,
        max_positions=4096,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        hdim = embed_dim # 5d^2
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = F.silu
        self.scaling = self.zdim ** -0.5 

        self.chunk_size = chunk_size
        self.norm = FusedRMSNorm(embed_dim, eps=1e-5)

        self.move = S6(embed_dim, dtype = torch.float32, device = "cuda")
        #MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)

        self.v_proj = nn.Linear(embed_dim, hdim)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size
        self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    # @torch.compile
    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) + bias
        if attn_mask is not None:
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = F.softmax(qk, dim=-1, dtype = torch.float).type_as(qk)
        return attn_weights

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """
        
        x = x.transpose(0, 1)
        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        dim = seq_len
        if 0 < self.chunk_size < seq_len:
            # assert dim % self.chunk_size == 0
            dim = self.chunk_size
        attn_mask = torch.triu(fill_with_neg_inf(torch.zeros(dim, dim)), 1).to(x)

        residual = x
        x = self.norm(x)
        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        mx = self.activation(self.move(x.type_as(v).transpose(0, 1))).transpose(0, 1)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
        # L x B x D
        u = torch.sigmoid(u)
        # L x B x (E+S)
        z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = z.unsqueeze(2) * self.gamma + self.beta
        # L x B x 2 x S -> L x B x S
        q, k = torch.unbind(z, dim=2)

        # L x B x D -> B x L x D
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)


        ctx_len = k.size(1)
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.chunk_size:
                q = q.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                q = q.reshape(bsz, nc, self.chunk_size, self.zdim)

            if ctx_len < self.chunk_size:
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
                k = k.reshape(bsz, nc, self.chunk_size, self.zdim)
                v = v.reshape(bsz, nc, self.chunk_size, self.hdim)
                if padding_mask is not None:
                    # B x L -> B x K x C
                    padding_mask = padding_mask.view(bsz, nc, self.chunk_size)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None


        attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        
        if before_attn_fn:
            return attn_weights, v

        # B x K x C x E -> B x L x E -> L x B x E
        h = torch.matmul(attn_weights, v).view(bsz, seq_len, self.hdim).transpose(0, 1)
        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        # L x B x D
        out = torch.addcmul(residual, u, h - residual)

        out = out.transpose(0, 1)
        return out