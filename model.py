from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 32758

    # RMSNorm param
    eps_norm: float = 1e-5

    # RoPE
    rope_theta: float = 500_000

    # KV cache
    max_seq_len: int = 32
    max_batch_size: int = 32


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


# TODO: Learn about RoPE here
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# TODO(@amindiro): implement
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm_x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class MLP(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.in_proj(x)
        x = F.gelu(x)
        x = self.out_proj(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.repeat_factor = config.n_heads // self.n_kv_heads

        self.wo = torch.nn.Linear(
            config.dim, config.n_heads * self.head_dim
        )  # (D, N_h*D_h)
        self.wk = torch.nn.Linear(
            config.dim, self.n_kv_heads * self.head_dim
        )  # (D, N_kv*D_h)

        self.wv = torch.nn.Linear(
            config.dim, self.n_kv_heads * self.head_dim
        )  # (D, N_kv*D_h)

        self.wo = torch.nn.Linear(
            self.n_heads * self.head_dim, config.dim
        )  # (D, N_kv*D_h)

        # TODO: maybe pass this as input?
        self.cache_k = torch.empty(
            (
                config.max_seq_len,
                config.max_batch_size,
                self.n_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.empty(
            (
                config.max_seq_len,
                config.max_batch_size,
                self.n_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        B, T, D = x.shape
        q = self.wo(x)  # (B,T,N_h*D_h)
        k = self.wk(x)  # (B,T,N_kv*D_h)
        v = self.wv(x)  # (B,T,N_kv*D_h)

        q = q.view(B, T, self.n_heads, self.head_dim)  # (B,T,N_h,D_h)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)  # (B,T,N_kv,D_h)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)  # (B,T,N_kv,D_h)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # Update kv_cache
        self.cache_k[:B, pos : pos + T] = k
        self.cache_v[:B, pos : pos + T] = v

        k = self.cache_k[:B, : pos + T]  # (B, cache_len + T, N_kv, D_h)
        v = self.cache_v[:B, : pos + T]  # (B, cache_len + T, N_kv, D_h)

        k = repeat_kv(k, self.repeat_factor)  # (B, cache_len + T, N_h, D_h)
        v = repeat_kv(v, self.repeat_factor)  # (B, cache_len + T, N_h, D_h)

        q = q.transpose(1, 2)  # (B, N_h, T, D_h)
        k = k.transpose(1, 2)  # (B, N_h, cache_len + T, D_h)
        v = v.transpose(1, 2)  # (B, N_h,cache_len + T, D_h)

        # TODO:: Triton kernel
        # sdpa implementation

        scores = torch.matmul(q, k.transpose(3, 2)) / math.sqrt(
            self.head_dim
        )  # (B, N_h, T, cache_len + T)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1)

        output = torch.matmul(scores, v)  # (B, N_h, T, D_h)
        output = output.transpose(2, 1)  # (B, T, N_h, D_h)

        # NOTE: contiguous before reshape/view here
        # use view here AFTER a contiguous call, faster and shouldn't throw
        return self.wo(output.contiguous().view(B, T, D))


class TransformerBlock(torch.nn.Module):
    def __init__(self, layer_idx: int, config: ModelConfig) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.dim, config.eps_norm)

        # NOTE: factor of 4 seems to be always here
        self.attention = Attention(config)
        self.mlp = MLP(config.dim, 4 * config.dim)
        self.mlp_norm = RMSNorm(config.dim, config.eps_norm)

    def forward(self, x, pos, freqs_cis, mask):
        x = x + self.attention(self.attn_norm(x), pos, freqs_cis, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size, config.dim)

        # transofmer's layers
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(self.config.n_layers):
            self.layers.append(TransformerBlock(layer_idx, config))

        # Output
        self.norm = RMSNorm(config.dim, eps=config.eps_norm)
        self.out = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPe setup
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta,
        )

    def forward(self, input_tokens: torch.Tensor, pos: int):
        _, S = input_tokens.shape
        x = self.embedding(input_tokens)
        mask = None

        # RoPE setup
        self.freqs_cis = self.freqs_cis.to(x.device)

        freqs_cis = self.freqs_cis[pos : pos + S]

        # Note: set the mask only on the prefill this is used to mask (upper triangular)
        if S > 1:
            mask = torch.full(
                (S, S), fill_value=float("-inf")
            )  #  -inf will be set 0 in softmax
            mask = torch.triu(mask, diagonal=1)
            # TODO:scores  matrice will be of shape : (S, S + cache_len). Maybe stack if prefill with kv cache
            # When performing KV caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            # mask = torch.hstack(
            #     [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            # ).type_as(h)
            mask = mask.type_as(x)

        for layer in self.layers:
            x = layer(x, pos, freqs_cis, mask)

        x = self.norm(x)
        out = self.out(x).float()

        return out
