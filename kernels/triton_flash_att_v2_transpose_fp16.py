"""

NOTE:
- only supported on ampere > https://github.com/woct0rdho/triton-windows/blob/readme/README.md


"""

import math
from typing import Any
import torch.nn.functional as F
import torch
import triton
import os
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PROFILE = int(os.getenv("PROFILE", 0)) == 1


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    S,
    stride_H,
    stride_k_d,
    stride_k_s,
    softmax_scale,
    D: tl.constexpr,
    Tc: tl.constexpr,
    Bc: tl.constexpr,
):
    # Get the thread index
    pid_y = tl.program_id(1)  # Head ID

    # Batch*Head offset
    batch_offset = pid_y * stride_H
    q_ptr = Q + batch_offset
    v_ptr = V + batch_offset
    o_ptr = O + batch_offset
    k_ptr = K + batch_offset

    pid_x = tl.program_id(0)  # Q Block ID

    # Q Offsets: Row-major
    # Rows: pid_x*Bc + 0..Bc
    # Cols: 0..D
    offs_m = pid_x * Bc + tl.arange(0, Bc)
    offs_d = tl.arange(0, D)

    # Q Pointer arithmetic
    offset_i = (offs_m[:, None] * D) + offs_d[None, :]
    mask_q = offs_m < S
    qi = tl.load(q_ptr + offset_i, mask=mask_q[:, None], other=0.0)

    # NOTE: for numerical stability bad idea to do softmax in fp16, bf16 could probably work but fp16 sucks
    prev_li = tl.zeros([Bc], dtype=tl.float32) + 1.0  # Init to 1.0 to avoid NaN
    prev_mi = tl.zeros([Bc], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Bc, D], dtype=tl.float32)

    # NOTE: Pre-scale Q to save muls inside loop
    qi = qi * softmax_scale
    qi = qi.to(tl.float16)

    for j in range(0, Tc):
        # K is physically (D, S). We want to load block (D, Bc). We need Rows 0..D and Cols j*Bc..+Bc
        current_cols = j * Bc + tl.arange(0, Bc)  # Iterate S dimension

        # Pointer Math: (Row_Idx * Stride_Row) + (Col_Idx * Stride_Col)
        # Row_Idx is offs_d (0..D)
        # Col_Idx is current_cols (S dimension)

        offset_j_k = (offs_d[:, None] * stride_k_d) + (
            current_cols[None, :] * stride_k_s
        )
        # Load K (D, Bc) directly!
        kj = tl.load(k_ptr + offset_j_k).to(tl.float16)

        # Load V (Bc,D)
        current_cols = j * Bc + tl.arange(0, Bc)  # Iterate S dimension
        offset_j_v = (current_cols[:, None] * D) + offs_d[None, :]
        vj = tl.load(v_ptr + offset_j_v).to(tl.float16)

        # NOTE: Cast to FP16 to trigger Tensor Cores
        # kj_fp16 = kj.to(tl.float16)
        Sij = tl.dot(qi, kj)

        # Softmax
        mij = tl.max(Sij, 1)
        pij = tl.exp(Sij - mij[:, None]).to(tl.float16)
        lij = tl.sum(pij, 1)

        # Update Stats
        mi_new = tl.maximum(prev_mi, mij)
        alpha = tl.exp(prev_mi - mi_new)
        beta = tl.exp(mij - mi_new)

        li_new = prev_li * alpha + lij * beta

        # Accumulate Output
        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(pij, vj)

        prev_li = li_new
        prev_mi = mi_new

    acc = acc / prev_li[:, None]

    # NOTE: accumulator in fp16
    tl.store(o_ptr + offset_i, acc.to(tl.float16), mask=mask_q[:, None])


def simple_attn(q, k, v):
    # Reference needs float for precision comparison
    att = q.float() @ k.float().transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v.float()
    return y.to(q.dtype)


def compute_sram_need(Br, Bc, D_h):
    device_properties = torch.cuda.get_device_properties(0)
    sram_needed = (3 * Br * D_h * 4) + (Bc * Br * 4)
    max_sram_size = device_properties.shared_memory_per_block
    print(f"Device Name: {device_properties.name}")
    print(f"Maximum Shared Memory (SRAM) Per Block: {max_sram_size} bytes")
    print(f"Shared Memory needed: {sram_needed} bytes")


def check_tma():
    # Print PTX to check for mma instructions
    print("\n=== Checking for Tensor Core (mma) instructions in PTX ===")
    #############################3
    # Access the compiled kernel from cache
    import glob
    import os as os_module

    cache_dir = os_module.path.expanduser("~/.triton/cache")
    ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
    if ptx_files:
        # Get most recent PTX file
        latest_ptx = max(ptx_files, key=os_module.path.getmtime)
        with open(latest_ptx, "r") as f:
            ptx_content = f.read()
        if "mma" in ptx_content:
            print("✓ Found mma instructions - Tensor Cores ARE being used!")
            for line in ptx_content.split("\n"):
                if "mma" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ No mma instructions found - Tensor Cores NOT being used")
            # Look for what dot product instructions are used
            print("\nLooking for fma/mul instructions:")
            for line in ptx_content.split("\n"):
                if "fma" in line.lower() or (
                    "mul" in line.lower() and "f16" in line.lower()
                ):
                    print(f"  {line.strip()}")
                    break
    else:
        print("No PTX files found in cache")
    #############################3


def main():
    B = 10
    S = 2048
    N_h = 64
    D_h = 32

    Br = Bc = 32
    Tc = S // Bc

    q = torch.randn(B, N_h, S, D_h, dtype=torch.float16).cuda()
    v = torch.randn(B, N_h, S, D_h, dtype=torch.float16).cuda()
    k = torch.randn(B, N_h, S, D_h, dtype=torch.float16).cuda()
    o = torch.zeros_like(q)

    # Transpose (feels like cheating)
    k_trans = k.transpose(-1, -2).contiguous()
    stride_k_d = k_trans.stride(2)
    stride_k_s = k_trans.stride(3)

    compute_sram_need(Br, Bc, D_h)

    print("=== profiling flash attention ===")

    # Grid: (Number of Q blocks, Batch * Heads)
    grid = (triton.cdiv(S, Bc), B * N_h)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        attn_kernel[grid](
            q,
            k_trans,
            v,
            o,
            S,
            q.stride(1),
            stride_k_d,
            stride_k_s,
            1 / math.sqrt(D_h),
            D_h,
            Tc,
            Bc,
            num_warps=4,  # or 8. Critical for Tensor Core parallelization
        )
        check_tma()

    if PROFILE:
        print("=== profiling reference simple attention ===")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # NOTE: reduce S or OOM
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CUDA]
    # ) as prof:
    #     o_simple = simple_attn(q, k, v)
    #
    # if PROFILE:
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #
    # assert torch.allclose(o, o_simple, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()
