import os
import math

from typing import Any
import torch.nn.functional as F
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

PROFILE = int(os.getenv("PROFILE",0)) ==1


# NOTE: (@aminediro): This is based on the FlashAttention1 paper
@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    stride_q_s, stride_q_d,  # Strides for Sequence and HeadDim
    stride_k_d, stride_k_s,  # Note: K is transposed, so we need stride for D and S
    stride_v_s, stride_v_d,
    stride_o_s, stride_o_d,
    l, m,
    S: tl.constexpr, D: tl.constexpr,
    Tc: tl.constexpr, Tr: tl.constexpr,
    Bc: tl.constexpr,
    softmax_scale,
):
    # We have Br threads for this program
    # Get the thread index
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # TODO: could do this using strides q.stride(1)
    # pass stride_q as arg
    batch_offset = (pid_x * tl.num_programs(1) * S * D) + (pid_y * S * D)

    k_ptr = K + batch_offset
    v_ptr = V + batch_offset
    q_ptr = Q + batch_offset
    o_ptr = O + batch_offset

    # TODO: pass stride_m
    # Stride for l,m is S.
    lm_batch_offset = (pid_x * tl.num_programs(1) * S) + (pid_y * S)
    l_ptr = l + lm_batch_offset
    m_ptr = m + lm_batch_offset

    offs_d = tl.arange(0, D)  # Iterate over D (Head Dim)
    offs_m = tl.arange(0, Bc) # Iterate over S (Rows of Q, O, L, M)
    offs_s = tl.arange(0, Bc) # Iterate over S (Cols of K, Rows V)

    # offset the batch*N_h, for each dim, skip to the next dim
    for j in range(0, Tc):
        # Load K_j, V_j from HBM to SRAM
        k_cols = j * Bc + offs_s
        # Pointer: Base + (Row_D * Stride_D) + (Col_S * Stride_S)
        k_ptrs = k_ptr + (offs_d[:, None] * stride_k_d) + (k_cols[None, :] * stride_k_s)
        kj = tl.load(k_ptrs) # Load (D, Bc)

        # Load Bc rows  from V
        v_rows = j * Bc + offs_s
        v_ptrs = v_ptr + (v_rows[:, None] * stride_v_s) + (offs_d[None, :] * stride_v_d)
        vj = tl.load(v_ptrs) # Load (Bc, D)

        # TODO: Run parallel loop
        for i in range(0, Tr):
            
            # Offsets for current Q block
            q_rows = i * Bc + offs_m
            
            # --- Load O, L, M (Accumulators) ---
            # O ptr: Base + (Row_S * Stride_S) + (Col_D * Stride_D)
            o_ptrs = o_ptr + (q_rows[:, None] * stride_o_s) + (offs_d[None, :] * stride_o_d)
            l_ptrs = l_ptr + q_rows
            m_ptrs = m_ptr + q_rows

            prev_oi = tl.load(o_ptrs)
            prev_li = tl.load(l_ptrs)
            prev_mi = tl.load(m_ptrs)

            # --- Load Q ---
            q_ptrs = q_ptr + (q_rows[:, None] * stride_q_s) + (offs_d[None, :] * stride_q_d)
            qi = tl.load(q_ptrs) # Load (Bc, D)

            # --- Computation ---
            # 1. Cast inputs to FP16 for Tensor Core HMMA
            # Shape: (Bc, D) x (D, Bc) -> (Bc, Bc)
            # qi_fp16 = qi.to(tl.float16)
            # kj_fp16 = kj.to(tl.float16)

            # 2. Dot Product
            Sij = tl.dot(qi, kj) * softmax_scale

            # 3. Softmax Logic
            mij = tl.max(Sij, 1) # Row max
            pij = tl.exp(Sij - mij[:, None])
            lij = tl.sum(pij, 1)

            # 4. Update Running Statistics
            mi_new = tl.maximum(prev_mi, mij)
            
            alpha = tl.exp(prev_mi - mi_new)
            beta = tl.exp(mij - mi_new)
            
            li_new = prev_li * alpha + lij * beta

            # 5. Update Output
            # Cast P and V to FP16 for second HMMA
            # pij_fp16 = pij.to(tl.float16)
            # vj_fp16 = vj.to(tl.float16)
            # Weighted sum
            
            oi_new = (
                alpha[:, None] * prev_li[:, None] * prev_oi
                + beta[:, None] * tl.dot(pij,vj)
            ) / li_new[:, None]

            # --- Store to HBM ---
            tl.store(o_ptrs, oi_new) # Write back O
            tl.store(m_ptrs, mi_new)
            tl.store(l_ptrs, li_new)


def simple_attn(q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def compute_sram_need(Br, Bc, D_h):
    device_properties = torch.cuda.get_device_properties(0)
    # NOTE:
    # (3 * Br * D_h * sizeof(float)) -> tile of q,k,v
    # (Br * Bc * sizeof(float)) -> tile scores (Br,Bc)
    sram_needed = (3 * Br * D_h * 4) + (Bc * Br * 4)
    max_sram_size = device_properties.shared_memory_per_block
    print(f"Device Name: {device_properties.name}")
    print(f"Maximum Shared Memory (SRAM) Per Block: {max_sram_size} bytes")
    print(f"Shared Memory needed: {sram_needed} bytes")


def main():
    B = 10
    N_h = 64
    S = 1024
    D_h = 16

    q = torch.randn(B, N_h, S, D_h).cuda()
    v = torch.randn(B, N_h, S, D_h).cuda()
    k = torch.randn(B, N_h, S, D_h).cuda()
    o = torch.zeros_like(q)

    l = torch.zeros(B, N_h, S).cuda()
    m = torch.full((B, N_h, S), float("-inf")).cuda()

    # flash attn block size
    Br = Bc = 32
    Tc = Tr = S // Bc

    compute_sram_need(Br, Bc, D_h)

    # NOTE:
    print("=== profiling flash attention ===")
    # simple_time = triton.testing.do_bench(lambda: attn_kernel[(B, N_h)](q, k, v, o, S, D_h, Tc, Tr, Bc, Br, 1/math.sqrt(D_h), l, m), rep = 400)
    # print(f"Flash attention : {simple_time*1000:.2f}us")
    k_trans = k.transpose(-1,-2).contiguous()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        attn_kernel[(B, N_h)](
        q, k_trans, v, o,
        q.stride(2), q.stride(3),
        k_trans.stride(2), k_trans.stride(3), # K strides: (stride_d, stride_s)
        v.stride(2), v.stride(3),       
        o.stride(2), o.stride(3),      
        l, m,
        S, D_h, Tc, Tr, Bc, 1 / math.sqrt(D_h)
    )
    if PROFILE :
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("=== profiling reference simple attention ===")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        o_simple = simple_attn(q, k, v)
    if PROFILE :
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    assert torch.allclose(o, o_simple, atol=1e-5, rtol=1e-5)



main()
