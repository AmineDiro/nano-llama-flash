import math

from typing import Any
import torch.nn.functional as F
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# NOTE: (@aminediro): This is based on the FlashAttention1 paper
#
@triton.autotune(
    configs=[
        triton.Config({"Bc": 16}, num_stages=3, num_warps=8),
        triton.Config({"Bc": 32}, num_stages=4, num_warps=1),
        triton.Config({"Bc": 32}, num_stages=4, num_warps=4),
    ],
    key=["D", "S"],
)
@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    S,
    stride_H,
    softmax_scale,
    D: tl.constexpr,
    Tc: tl.constexpr,
    Bc: tl.constexpr,
):
    # Get the thread index
    pid_y = tl.program_id(1)
    batch_offset = pid_y * stride_H # skip batch,head offset: idx*B*N_h
    q_ptr = Q + batch_offset
    k_ptr = K + batch_offset
    v_ptr = V + batch_offset
    o_ptr = O + batch_offset

    # Loading Query block (Bc,D)
    pid_x = tl.program_id(0)
    S_i_offset = pid_x * Bc * D + tl.arange(0, Bc) * D  # D is the stride
    offset_i = (S_i_offset)[:, None] + tl.arange(0, D)[None, :]
    qi = tl.load(q_ptr + offset_i)  # shape (Br,Br) == (Bc,Bc)

    # Block accumulator and running max
    prev_li = tl.zeros([Bc], dtype=tl.float32)
    prev_mi = tl.zeros([Bc], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Bc, D], dtype=tl.float32)

    # offset the batch*N_h, for each dim, skip to the next dim
    for j in range(0, Tc):
        # Load K_j, V_j from HBM to SRAM
        # NOTE: STUPID mistake! the stride
        offset_j = (j * Bc + tl.arange(0, Bc))[:, None] * D + tl.arange(0, D)[None, :]
        kj = tl.load(k_ptr + offset_j)  # shape(Bc,Bc)
        vj = tl.load(v_ptr + offset_j)  # shape(Bc,Bc)

        # TODO: Run parallel loop
        # Compute Sij on Chip Q_i * K_j.T / sqrt(D_h)
        Sij = tl.dot(qi, tl.trans(kj)) * softmax_scale  # (Bc,Br) == (Bc,Bc)

        # Rowmax(Sij): (Bc,)
        mij = tl.max(Sij, 1)
        pij = tl.exp(Sij - mij[:, None])  # (Bc,Bc)
        lij = tl.sum(pij, 1)  # (Bc,)

        # Running maximum
        mi_new = tl.maximum(prev_mi, mij)

        # Compute scaling factors using previous_max
        alpha = tl.exp(prev_mi - mi_new)
        beta = tl.exp(mij - mi_new)

        # Update running sum
        li_new = prev_li * alpha + lij * beta

        # Update the output block
        acc = (
            alpha[:, None] * prev_li[:, None] * acc
            + beta[:, None] * tl.dot(pij, vj)
        ) / li_new[:, None]

        prev_li = li_new
        prev_mi = mi_new
    
    # Update in HBM
    tl.store(o_ptr + offset_i, acc)
    
    # TODO: maybe we need to store this for the backprop
    # tl.store(m_ptr + S_i_offset, mi_new)
    # tl.store(l_ptr + S_i_offset, li_new)


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
    N_h = 32
    S = 64
    D_h = 128

    q = torch.randn(B, N_h, S, D_h).cuda()
    v = torch.randn(B, N_h, S, D_h).cuda()
    k = torch.randn(B, N_h, S, D_h).cuda()
    o = torch.zeros_like(q)

    l = torch.zeros(B, N_h, S).cuda()
    m = torch.full((B, N_h, S), float("-inf")).cuda()

    # flash attn block size
    Br = Bc = 32
    Tc = Tr = S // Bc  # NOTE: S%Br == 0

    compute_sram_need(Br, Bc, D_h)

    # NOTE:
    print("=== profiling flash attention ===")
    # simple_time = triton.testing.do_bench(lambda: attn_kernel[(B, N_h)](q, k, v, o, S, D_h, Tc, Tr, Bc, Br, 1/math.sqrt(D_h), l, m), rep = 400)
    # print(f"Flash attention : {simple_time*1000:.2f}us")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        grid = lambda META: (triton.cdiv(S, META["Bc"]), B * N_h)
        attn_kernel[grid](
            q, k, v, o, S, q.stride(1), 1 / math.sqrt(D_h), D_h, Tc
        )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("=== profiling reference simple attention ===")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        o_simple = simple_attn(q, k, v)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    assert torch.allclose(o, o_simple, atol=1e-6, rtol=1e-6)


main()
