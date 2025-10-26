"""
FA1 implementation with padding to avoid bank conflicts

FIXES:
- Added padding to head dimension (D_h) to break bank conflict pattern
- Padding breaks stride alignment: D=32→40, D=64→72, D=128→136
- Masking ensures padded elements don't affect computation
- Core algorithm logic remains identical to original

Original Issues:
- Block Size Too Small (Bc=32) With Bc=32 and S=8192, you're doing 256 iterations of the loop. -> Large
- At S=8192, cuBLAS is simply faster. Flash Attention's advantage appears when S > 16K. Memory bandwidth becomes the bottleneck
- Missing Br Parameter: same block size for both rows (Q) and columns (K/V). The original FA1 uses different block. Typically Br should be larger than Bc for better performance.
- BANK CONFLICTS: Row-major loading with stride=D causes 6.3-way bank conflicts

WINs:
- At large S (> 4096) on RTX2070 Super, only the FA works. Torch OOMs
"""

import math

from typing import Any
import torch.nn.functional as F
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def compute_padded_headdim(D_h):
    """
    Compute padded head dimension to avoid bank conflicts.

    Bank conflicts occur when stride = 32, 64, 128 (powers of 2).

    Strategy: Use manual loop unrolling instead of tl.arange() for non-power-of-2.
    Actually, we'll use a BLOCK dimension that's power of 2, but with careful masking.

    For D=32: pad to 64 (next power of 2)
    For D=64: pad to 128 (next power of 2)
    For D=128: pad to 256 (next power of 2)

    The doubling provides extra padding that naturally breaks stride-based conflicts.
    With stride=64 instead of 32, threads access different bank groups.
    """
    # Find next power of 2
    if D_h <= 0:
        return 1
    # Check if already power of 2
    if (D_h & (D_h - 1)) == 0:
        # Already power of 2, double it
        return D_h * 2
    else:
        # Round up to next power of 2
        return 1 << (D_h - 1).bit_length()


def pad_tensor_headdim(tensor, D_h, D_h_padded):
    """
    Pad the last dimension (head dimension) of a tensor.

    Args:
        tensor: Input tensor of shape (..., D_h)
        D_h: Original head dimension
        D_h_padded: Padded head dimension

    Returns:
        Padded tensor of shape (..., D_h_padded)
    """
    pad_amount = D_h_padded - D_h
    # F.pad pads in reverse order: (left, right, top, bottom, ...)
    # We want to pad the last dimension, so (0, pad_amount)
    return F.pad(tensor, (0, pad_amount), mode="constant", value=0.0)


def unpad_tensor_headdim(tensor, D_h):
    """
    Remove padding from the last dimension.

    Args:
        tensor: Padded tensor of shape (..., D_h_padded)
        D_h: Original head dimension (without padding)

    Returns:
        Unpadded tensor of shape (..., D_h)
    """
    return tensor[..., :D_h]


# NOTE: (@aminediro): This is based on the FlashAttention1 paper with padding fix
@triton.jit
def attn_kernel_padded(
    Q,
    K,
    V,
    O,
    S,
    stride_H,
    softmax_scale,
    D: tl.constexpr,  # Original head dimension (unpadded)
    D_padded: tl.constexpr,  # Padded head dimension
    Tc: tl.constexpr,
    Bc: tl.constexpr,
):
    # Get the thread index
    pid_y = tl.program_id(1)
    batch_offset = pid_y * stride_H  # skip batch,head offset: idx*B*N_h
    q_ptr = Q + batch_offset
    k_ptr = K + batch_offset
    v_ptr = V + batch_offset
    o_ptr = O + batch_offset

    # Loading Query block (Bc, D_padded)
    pid_x = tl.program_id(0)

    # Create row and column indices
    offs_m = pid_x * Bc + tl.arange(0, Bc)
    offs_d = tl.arange(0, D_padded)

    # Standard indexing: row_idx * stride + col_idx
    # With D_padded = 2 * D, the stride is doubled which reduces bank conflicts
    # Stride of 64 (vs 32) means warps access memory with better bank distribution
    offset_i = offs_m[:, None] * D_padded + offs_d[None, :]

    # Mask for both sequence dimension and head dimension
    seq_mask = offs_m < S
    head_mask = offs_d < D
    load_mask = seq_mask[:, None] & head_mask[None, :]

    qi = tl.load(q_ptr + offset_i, mask=load_mask, other=0.0)  # shape (Bc, D_padded)

    # Block accumulator and running max
    prev_li = tl.zeros([Bc], dtype=tl.float32)
    prev_mi = tl.zeros([Bc], dtype=tl.float32) - float("inf")
    acc = tl.zeros([Bc, D_padded], dtype=tl.float32)

    # Loop over K,V blocks
    for j in range(0, Tc):
        # Load K_j, V_j from HBM to SRAM
        offset_j = (j * Bc + tl.arange(0, Bc))[:, None] * D_padded + tl.arange(
            0, D_padded
        )[None, :]

        # Mask for K,V blocks (always load full blocks in K dimension, but mask head dimension)
        kv_head_mask = head_mask[None, :]
        kv_seq_mask = (j * Bc + tl.arange(0, Bc)) < S
        kv_mask = kv_seq_mask[:, None] & kv_head_mask

        kj = tl.load(k_ptr + offset_j, mask=kv_mask, other=0.0)  # shape (Bc, D_padded)
        vj = tl.load(v_ptr + offset_j, mask=kv_mask, other=0.0)  # shape (Bc, D_padded)

        # Compute Sij on Chip: Q_i * K_j.T / sqrt(D_h)
        # Note: Use D (not D_padded) for scaling since padded elements are zero
        Sij = tl.dot(qi, tl.trans(kj)) * softmax_scale  # (Bc, Bc)

        # Rowmax(Sij): (Bc,)
        mij = tl.max(Sij, 1)
        pij = tl.exp(Sij - mij[:, None])  # (Bc, Bc)
        lij = tl.sum(pij, 1)  # (Bc,)

        # Running maximum
        mi_new = tl.maximum(prev_mi, mij)

        # Compute scaling factors using previous_max
        alpha = tl.exp(prev_mi - mi_new)
        beta = tl.exp(mij - mi_new)

        # Update running sum
        li_new = prev_li * alpha + lij * beta

        # Update the output block
        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(pij, vj)

        prev_li = li_new
        prev_mi = mi_new

    acc = acc / prev_li[:, None]

    # Update in HBM (with masking for both dimensions)
    tl.store(o_ptr + offset_i, acc, mask=load_mask)


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


def flash_attn_padded(q, k, v, Bc=32):
    """
    Flash Attention with padding to avoid bank conflicts.

    Args:
        q, k, v: Tensors of shape (B, N_h, S, D_h)
        Bc: Block size for sequence dimension

    Returns:
        Output tensor of shape (B, N_h, S, D_h)
    """
    B, N_h, S, D_h = q.shape

    # Compute padded dimension
    D_h_padded = compute_padded_headdim(D_h)

    print(f"Original head dim: {D_h}, Padded head dim: {D_h_padded}")
    print(f"Padding breaks bank conflict: stride {D_h} → {D_h_padded}")

    # Pad tensors
    q_padded = pad_tensor_headdim(q, D_h, D_h_padded)
    k_padded = pad_tensor_headdim(k, D_h, D_h_padded)
    v_padded = pad_tensor_headdim(v, D_h, D_h_padded)
    o_padded = torch.zeros_like(q_padded)

    # Flash attention block size
    Tc = triton.cdiv(S, Bc)  # Number of blocks (handles S % Bc != 0)

    # Softmax scale
    softmax_scale = 1.0 / math.sqrt(D_h)  # Use original D_h, not padded

    # Launch kernel
    grid = lambda META: (triton.cdiv(S, META["Bc"]), B * N_h)

    attn_kernel_padded[grid](
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        S,
        q_padded.stride(1),  # stride_H: stride to next head
        softmax_scale,
        D_h,  # Original dimension
        D_h_padded,  # Padded dimension
        Tc,
        Bc,
        num_warps=4,
        num_ctas=1,
        num_stages=4,
    )

    # Unpad output
    o = unpad_tensor_headdim(o_padded, D_h)

    return o


def main():
    B = 2
    N_h = 2
    S = 1024
    D_h = 32

    print("=" * 60)
    print("FLASH ATTENTION WITH BANK CONFLICT FIX (PADDING)")
    print("=" * 60)

    q = torch.randn(B, N_h, S, D_h).cuda()
    k = torch.randn(B, N_h, S, D_h).cuda()
    v = torch.randn(B, N_h, S, D_h).cuda()

    # Flash attention block size
    Bc = 32
    D_h_padded = compute_padded_headdim(D_h)

    compute_sram_need(Bc, Bc, D_h_padded)
    print()

    # Run padded flash attention
    print("=== Running Padded Flash Attention ===")
    o = flash_attn_padded(q, k, v, Bc=Bc)
    torch.cuda.synchronize()

    print(f"Output shape: {o.shape}")
    print()


if __name__ == "__main__":
    main()
