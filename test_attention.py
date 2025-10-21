import torch
from model import Attention, ModelConfig, Transformer, precompute_freqs_cis


def test_attn():
    assert torch.cuda.is_available()

    config = ModelConfig()
    attn = Attention(config)

    device = torch.device(0)
    attn = attn.to(device)

    B, T, D = 1, 3, config.dim
    pos = 0
    x = torch.rand(B, T, D).to(device)
    freqs = precompute_freqs_cis(
        config.dim // config.n_heads,
        4,
        config.rope_theta,
    ).to(device)

    res = attn(x, pos, freqs[pos : pos + T], None)

    assert len(res.shape) == 3
    assert res.shape == (B, T, D)


def test_transformer():
    assert torch.cuda.is_available()
    device = torch.device(0)

    config = ModelConfig()
    model = Transformer(config).to(device)

    B, T, D = 1, 3, config.dim
    pos = 0
    x = torch.rand(B, T, D).to(device)
    freqs = precompute_freqs_cis(
        config.dim // config.n_heads,
        4,
        config.rope_theta,
    ).to(device)

    res = model(x, pos)

    assert len(res.shape) == 3
    assert res.shape == (B, T, D)
