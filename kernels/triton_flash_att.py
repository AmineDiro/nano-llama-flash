import math
from typing import Any

import torch.nn.functional as F
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# NOTE: (@aminediro): This is based on the FlashAttention1 paper
@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    S,
    D: tl.constexpr,
    Tc: tl.constexpr,
    Tr: tl.constexpr,
    Bc: tl.constexpr,
    Br: tl.constexpr,
    softmax_scale,
    l,
    m,
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

    l_ptr = l + batch_offset
    m_ptr = m + batch_offset

    # Create in SRAM
    li = tl.zeros([Bc], dtype=tl.float32)
    oi = tl.zeros([Bc, D], dtype=tl.float32)
    mi = tl.full([Bc], value = float('-inf'), dtype=tl.float32)

    # offset the batch*N_h, for each dim, skip to the next dim
    for j in range(0, Tc):
        # Load K_j, V_j from HBM to SRAM
        offset_j = (j * Bc + tl.arange(0, Bc))[:,None] + tl.arange(0, D)[None,:]
        kj = tl.load(k_ptr + offset_j)  # shape(Bc,Bc)
        vj = tl.load(v_ptr + offset_j)  # shape(Bc,Bc)

        for i in range(0, Tr):
            # Load Q_i, O_i, l_i, m_i from HBM to SRAM
            S_i_offset = i * Bc + tl.arange(0, Bc)
            offset_i = (S_i_offset)[:,None] + tl.arange(0, D)[None,:]

            qi = tl.load(q_ptr + offset_i)  # shape (Br,Br) == (Bc,Bc)

            # Compute Sij on Chip Q_i * K_j.T / sqrt(D_h)
            Sij = tl.dot(qi, tl.trans(kj)) * softmax_scale  # (Bc,Br) == (Bc,Bc)

            # Rowmax(Sij): (Bc,)
            mij = tl.max(Sij, 1)
            pij = tl.exp(Sij - mij[:None]) # (Bc,Bc)
            lij = tl.sum(pij, 1) # (Bc,)

            # Running maximum
            mi_new = tl.maximum(mi, mij)

            # Update the running log-sum-exp and max
            alpha = tl.exp(mi - mi_new)
            beta = tl.exp(mij - mi_new)
            li_new = li * alpha + lij * beta

            # li norm 
            # li_new_diag = tl.eye(Bc, dtype=li_new.dtype) * li_new[:, None]
            # li_diag = tl.eye(Bc, dtype=li.dtype) * li_new[:, None]
            # li_norm = li_diag / li_new_diag
            oi = (alpha[:,None]*li[:,None] * oi + beta[:,None] * tl.dot(pij, vj)) / li_new[:,None]

            # Update moments in HBM
            mi = mi_new
            li = li_new

            # Update in HBM
            tl.store(o_ptr + offset_i, oi)
            tl.store(m_ptr + S_i_offset, mi)
            tl.store(l_ptr + S_i_offset, li)

def simple_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def compute_sram_need(Br,Bc,D_h):
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

    B = 1  # 16
    N_h = 2  # 12
    S = 64
    D_h = 64

    q = torch.randn(B, N_h, S, D_h).cuda()
    v = torch.randn(B, N_h, S, D_h).cuda()
    k = torch.randn(B, N_h, S, D_h).cuda()
    o = torch.zeros_like(q)

    l = torch.zeros(B, N_h, S).cuda()
    m = torch.full((B, N_h, S), float("-inf")).cuda()

    # flash attn block size
    Br = Bc = 32
    Tc = Tr = S // Bc

    compute_sram_need(Br,Bc,D_h) 

    # NOTE: 
    attn_kernel[(B, N_h)](q, v, k, o, S, D_h, Tc, Tr, Bc, Br, 1/math.sqrt(D_h), l, m)
 
    o_simple= simple_attn(q,k,v)

    # Check if the results are the same
    print(o[0,0,0,:])
    print(o_simple[0,0,0,:])
    assert torch.allclose(o, o_simple, atol=1e-6)


main()
