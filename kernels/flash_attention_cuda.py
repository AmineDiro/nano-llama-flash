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
    name="flash_attention_cuda",
    sources=[
        os.path.join(current_dir, "flash_attention_wrapper.cpp"),
        os.path.join(current_dir, "flash_attention_kernel.cu"),
    ],
    extra_cuda_cflags=[
        "-O2",  # Basic optimization
        "--use_fast_math",  # Fast math operations
    ],
    verbose=True,
)
print("CUDA extension compiled successfully!")


def flash_attention(q, k, v, Bc=32):
    """
    Flash Attention forward pass

    Args:
        q: Query tensor of shape (B, N_h, S, D_h)
        k: Key tensor of shape (B, N_h, S, D_h)
        v: Value tensor of shape (B, N_h, S, D_h)
        Bc: Block size (default: 32)

    Returns:
        Output tensor of shape (B, N_h, S, D_h)
    """
    # Validate inputs
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
    assert q.dtype == torch.float32, "Only float32 is supported"

    B, N_h, S, D_h = q.shape
    assert S % Bc == 0, f"Sequence length {S} must be divisible by block size {Bc}"

    output = cuda_module.flash_attn_forward(q, k, v, B, N_h, S, D_h, Bc)

    return output


def simple_attn(q, k, v):
    """Reference attention implementation"""
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def compute_sram_need(Br, Bc, D_h):
    """Calculate SRAM requirements for the kernel"""
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
    """Test function comparing CUDA implementation with reference"""
    print("\n" + "=" * 50)
    print("Testing Flash Attention CUDA Implementation")
    print("=" * 50)

    # Test parameters
    B = 2
    N_h = 4
    S = 32
    D_h = 128
    Bc = 32

    print(f"\nTest configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Number of heads (N_h): {N_h}")
    print(f"  Sequence length (S): {S}")
    print(f"  Head dimension (D_h): {D_h}")
    print(f"  Block size (Bc): {Bc}")

    # Check SRAM requirements
    print(f"\nChecking SRAM requirements...")
    if not compute_sram_need(Bc, Bc, D_h):
        print("WARNING: SRAM requirements may exceed device limits!")

    # Create random inputs
    print(f"\nCreating test tensors...")
    q = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)

    # Run CUDA kernel
    print(f"\nRunning CUDA Flash Attention kernel...")
    torch.cuda.synchronize()
    o_cuda = flash_attention(q, k, v, Bc)
    torch.cuda.synchronize()
    print(f"✓ CUDA kernel completed")

    # Run reference implementation
    print(f"\nRunning reference PyTorch attention...")
    o_ref = simple_attn(q, k, v)
    torch.cuda.synchronize()
    print(f"✓ Reference computation completed")

    # Check correctness with tolerance
    atol = rtol = 1e-5
    is_close = torch.allclose(o_cuda, o_ref, atol=atol, rtol=rtol)

    if is_close:
        print(f"Results match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"Results DO NOT match within tolerance!")
        print(f"  First few values (CUDA): {o_cuda.flatten()[:5]}")
        print(f"  First few values (Ref):  {o_ref.flatten()[:5]}")

    print("\n" + "=" * 50)
    return is_close


def benchmark_flash_attention():
    """Benchmark CUDA kernel performance"""
    print("\n" + "=" * 50)
    print("Benchmarking Flash Attention")
    print("=" * 50)

    B = 4
    N_h = 8
    D_h = 64
    Bc = 32

    sequence_lengths = [512, 1024, 2048, 4096]

    print(f"\nConfiguration: B={B}, N_h={N_h}, D_h={D_h}, Bc={Bc}")
    print(
        f"\n{'Seq Length':>12} | {'CUDA (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>10}"
    )
    print("-" * 60)

    for S in sequence_lengths:
        if S % Bc != 0:
            continue

        q = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
        k = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)
        v = torch.randn(B, N_h, S, D_h, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _ = flash_attention(q, k, v, Bc)
            _ = simple_attn(q, k, v)
        torch.cuda.synchronize()

        # Benchmark CUDA
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            _ = flash_attention(q, k, v, Bc)
        end.record()
        torch.cuda.synchronize()
        cuda_time = start.elapsed_time(end) / 10

        # Benchmark PyTorch
        start.record()
        for _ in range(10):
            _ = simple_attn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / 10

        speedup = pytorch_time / cuda_time
        print(
            f"{S:>12} | {cuda_time:>12.3f} | {pytorch_time:>12.3f} | {speedup:>10.2f}x"
        )

    print("=" * 60)


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
