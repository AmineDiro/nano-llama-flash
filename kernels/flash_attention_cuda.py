import math
import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile the CUDA extension
print("Compiling CUDA extension...")
cuda_module = load(
    name="flash_attention_cuda_v2",
    sources=[
        os.path.join(current_dir, "flash_attention_wrapper.cpp"),
        os.path.join(current_dir, "flash_attention_kernel.cu"),
    ],
    extra_cuda_cflags=[
        "-O2",
        "--use_fast_math",
        "-lineinfo",  # Line-level profiling info for NCU
        "-g",  # Debug symbols
    ],
    verbose=True,
)
print("CUDA extension compiled successfully!")


def flash_attention(q, k, v, Bc=32):
    # Validate inputs
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
    assert q.dtype == torch.float32, "Only float32 is supported"

    assert q.size(2) % Bc == 0, (
        f"Sequence length {q.size(2)} must be divisible by block size {Bc}"
    )

    output = cuda_module.flash_attn_forward(q, k, v, Bc)

    return output


def simple_attn(q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def compute_sram_need(Br, Bc, D_h):
    device_properties = torch.cuda.get_device_properties(0)
    # (3 * Br * D_h * sizeof(float)) -> tile of q,k,v
    # (Br * Bc * sizeof(float)) -> tile scores (Br,Bc)
    sram_needed = (3 * Br * D_h * 4) + (Bc * Br * 4)
    max_sram_size = device_properties.shared_memory_per_block
    print(f"Device Name: {device_properties.name}")
    print(f"Maximum Shared Memory (SRAM) Per Block: {max_sram_size} bytes")
    print(f"Shared Memory needed: {sram_needed} bytes")
    return sram_needed <= max_sram_size


def test_flash_attention():
    print("Testing Flash Attention CUDA Implementation")

    # Test parameters
    B = 1
    N_h = 16
    S = 512
    D_h = 33
    Bc = 32

    # Check SRAM requirements
    print(f"\nChecking SRAM requirements...")
    if not compute_sram_need(Bc, Bc, D_h):
        print("WARNING: SRAM requirements may exceed device limits!")

    q = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)

    # Run CUDA kernel
    print(f"\nRunning CUDA Flash Attention kernel...")
    torch.cuda.synchronize()
    o_cuda = flash_attention(q, k, v, Bc)
    torch.cuda.synchronize()

    # Run reference implementation
    print(f"\nRunning reference PyTorch attention...")
    o_ref = simple_attn(q, k, v)
    torch.cuda.synchronize()

    # Check correctness with tolerance
    atol = rtol = 1e-5
    is_close = torch.allclose(o_cuda, o_ref, atol=atol, rtol=rtol)

    if is_close:
        print(f"Results match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"Results DO NOT match within tolerance!")
        print(f"  First few values (CUDA): {o_cuda.flatten()[:5]}")
        print(f"  First few values (Ref):  {o_ref.flatten()[:5]}")

    return is_close


def main():
    """Main function - run tests"""
    # Test correctness
    success = test_flash_attention()

    # if success:
    #     # Run benchmark if test passes
    #     benchmark_flash_attention()
    # else:
    #     print("\n⚠️  Skipping benchmark due to correctness test failure")
    #


if __name__ == "__main__":
    main()
