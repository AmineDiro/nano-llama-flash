import torch
import triton
from model import Attention, ModelConfig, Transformer, precompute_freqs_cis

from triton_add_kernel import add_kernel

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_attn():
    assert torch.cuda.is_available()

    attn = Attention(config)
    config = ModelConfig()

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

    config = ModelConfig(n_layers=2)
    model = Transformer(config).to(device)

    B, T = (1, 10)
    pos = 0
    x = torch.randint(0, config.vocab_size, (B, T), dtype=torch.int32).to(device)
    res = model(x, pos)

    assert len(res.shape) == 3
    assert res.shape == (B, T, config.vocab_size)


def test_triton_kernel():
    M = 1024
    N = 100 * 1024  # Size of the arrays (1M elements)
    BLOCK_SIZE = 1024
    A = torch.randn((M, N), device="cuda", dtype=torch.float32)
    B = torch.randn((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    # Strides for array access
    stride_A = stride_B = stride_C = 1

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(M * N, meta["BLOCK_SIZE"]),)  # noqa: E731
    add_kernel[grid](A, B, C, M * N, stride_A, stride_B, stride_C, BLOCK_SIZE)

    # Verify the result using PyTorch
    C_pytorch = A + B

    # Check if the results are the same
    assert torch.allclose(C, C_pytorch, atol=1e-6)
