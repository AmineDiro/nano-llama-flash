"""
FA1 implementation
Issues:
- Block Size Too Small (Bc=32) With Bc=32 and S=8192, you're doing 256 iterations of the loop. -> Large
- At S=8192, cuBLAS is simply faster. Flash Attention's advantage appears when S > 16K. Memory bandwidth becomes the bottleneck
- Missing Br Parameter: same block size for both rows (Q) and columns (K/V). The original FA1 uses different block. Typically Br should be larger than Bc for better performance.

WINs:
- At large S (> 4096) on RTX2070 Super, only the FA works. Torch OOMs
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
    Q, K, V, O,
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
    pid_y = tl.program_id(1) # Head ID
    
    # Batch*Head offset 
    batch_offset = pid_y * stride_H  
    q_ptr = Q + batch_offset
    v_ptr = V + batch_offset
    o_ptr = O + batch_offset
    k_ptr = K + batch_offset

    pid_x = tl.program_id(0) # Q Block ID
    
    # Q Offsets: Row-major
    # Rows: pid_x*Bc + 0..Bc
    # Cols: 0..D
    offs_m = pid_x * Bc + tl.arange(0, Bc)
    offs_d = tl.arange(0, D)
    
    # Q Pointer arithmetic
    offset_i = (offs_m[:, None] * D) + offs_d[None, :]
    mask_q = offs_m < S
    qi = tl.load(q_ptr + offset_i, mask=mask_q[:, None], other=0.0)
    
    prev_li = tl.zeros([Bc], dtype=tl.float32)
    prev_mi = tl.zeros([Bc], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Bc, D], dtype=tl.float32)

    # NOTE: Pre-scale Q to save muls inside loop
    qi = qi * softmax_scale

    for j in range(0, Tc):
        # K is physically (D, S). We want to load block (D, Bc). We need Rows 0..D and Cols j*Bc..+Bc
        current_cols = j * Bc + tl.arange(0, Bc) # Iterate S dimension
        
        # Pointer Math: (Row_Idx * Stride_Row) + (Col_Idx * Stride_Col)
        # Row_Idx is offs_d (0..D)
        # Col_Idx is current_cols (S dimension)

        offset_j_k = (offs_d[:, None] * stride_k_d) + (current_cols[None, :] * stride_k_s)
        # Load K (D, Bc) directly!
        kj = tl.load(k_ptr + offset_j_k)

        # V is standard (S, D). We load (Bc, D)
        # Rows j*Bc..+Bc, Cols 0..D
        offset_j_v = (current_cols[:, None] * D) + offs_d[None, :]
        vj = tl.load(v_ptr + offset_j_v)
        Sij = tl.dot(qi, kj) # no transpose need for kj

        # Softmax
        mij = tl.max(Sij, 1)
        pij = tl.exp(Sij - mij[:, None])
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

    # Final Normalization with the correct sum
    acc = acc / prev_li[:, None]
    
    # Update in HBM
    # tl.store(o_ptr + offset_i, acc.to(tl.float16), mask=mask_q[:, None])
    tl.store(o_ptr + offset_i, acc, mask=mask_q[:, None])


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


def main():
    B = 10
    N_h = 64
    S = 1024
    # TODO: change to 64, 16 too small for TC
    D_h = 32

    q = torch.randn(B, N_h, S, D_h).cuda()
    v = torch.randn(B, N_h, S, D_h).cuda()
    k = torch.randn(B, N_h, S, D_h).cuda()
    o = torch.zeros_like(q)

    
    # Transpose (feels like cheating)
    k_trans = k.transpose(-1, -2).contiguous()
    stride_k_d = k_trans.stride(2) 
    stride_k_s = k_trans.stride(3)

    Br = Bc = 32
    Tc = S // Bc

    compute_sram_need(Br, Bc, D_h)

    print("=== profiling flash attention ===")
    
    # Grid: (Number of Q blocks, Batch * Heads)
    grid = (triton.cdiv(S, Bc), B * N_h) 
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
    )

    if PROFILE :
        print("=== profiling reference simple attention ===")
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        o_simple = simple_attn(q, k, v)
    if PROFILE :

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    assert torch.allclose(o, o_simple, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    main()
