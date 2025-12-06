"""
Minimal matmul kernel to test if Triton generates tensor core (mma) instructions.
Based on official Triton tutorial pattern.
"""
import torch
import triton
import triton.language as tl
import glob
import os as os_module


def check_mma():
    cache_dir = os_module.path.expanduser("~/.triton/cache")
    ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
    print("\n=== Checking for Tensor Core (mma) instructions in PTX ===")
    if ptx_files:
        # Get most recent PTX file
        latest_ptx = max(ptx_files, key=os_module.path.getmtime)
        with open(latest_ptx, 'r') as f:
            ptx_content = f.read()
        if "mma" in ptx_content:
            print("✓ Found mma instructions - Tensor Cores ARE being used!")
            for line in ptx_content.split('\n'):
                if 'mma' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ No mma instructions found - Tensor Cores NOT being used")
            # Look for what dot product instructions are used
            print("\nLooking for fma/mul instructions:")
            for line in ptx_content.split('\n'):
                if 'fma' in line.lower() or ('mul' in line.lower() and 'f16' in line.lower()):
                    print(f"  {line.strip()}")
                    break
    else:
        print("No PTX files found in cache")


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)

        # tl.dot should trigger tensor cores
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def main():
    print("=== Tensor Core Test ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    M, N, K = 128, 128, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    print(f"\nMatrix: {M}x{K} @ {K}x{N}")
    print(f"Block sizes: M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}")

    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    C = torch.zeros(M, N, dtype=torch.float16, device='cuda')

    grid = (M // BLOCK_M, N // BLOCK_N)

    print(f"\nLaunching kernel with grid={grid}, num_warps=4...")
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()

    C_ref = A @ B
    if torch.allclose(C, C_ref, atol=1e-2, rtol=1e-2):
        print("✓ Results match reference!")
    else:
        print(f"✗ Results don't match. Max diff: {(C - C_ref).abs().max().item()}")

    check_mma()


if __name__ == "__main__":
    main()
