"""
Fused Attention (Simplified Forward Pass)
=========================================

This is a simplified Triton implementation of the Flash Attention v2 algorithm's
forward pass, adapted from the original work by Tri Dao.

This version is stripped down to the bare essentials for a typical CUDA GPU
(e.g., NVIDIA RTX 2070), removing hardware-specific code for AMD, Hopper,
and Blackwell architectures, as well as support for FP8 data types.
The backward pass has also been removed to focus solely on the forward computation.

Credits:
- Tri Dao (Original Flash Attention v2 algorithm: https://tridao.me/publications/flash2/flash2.pdf)
- OpenAI kernel team (Original Triton implementation)
"""

import torch
import triton
import triton.language as tl

# --- Triton Kernels ---


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k,
    v,
    sm_scale,
    start_m,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    """
    Inner loop for the forward pass of attention.

    Computes a block of the attention output. This version is simplified for
    causal attention.
    """
    # Define the range of columns to iterate over for the K and V matrices.
    # Causal attention means we only attend to tokens up to the current one.
    if STAGE == 1:
        # Process blocks before the current diagonal block.
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Process the diagonal block.
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:  # Unused, but kept for structural clarity
        lo, hi = 0, N_CTX

    # Pointers to K and V matrices, offset to the current block.
    k_ptrs = (
        k + lo * HEAD_DIM + offs_n[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    )
    v_ptrs = (
        v + lo * HEAD_DIM + offs_n[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    )

    # Loop over blocks of K and V.
    for start_n in range(lo, hi, BLOCK_N):
        # -- Load K and V --
        k_block = tl.load(k_ptrs)
        v_block = tl.load(v_ptrs)

        # -- Compute QK^T --
        qk = tl.dot(q, tl.trans(k_block))

        # -- Apply Causal Mask (if in the diagonal block) --
        if STAGE == 2:
            # Create a mask to zero out attention to future tokens.
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * sm_scale + tl.where(mask, 0, -1.0e6)

        else:
            qk = qk * sm_scale

        # Update running max for numerical stability.
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)

        # -- Update Statistics and Accumulator --
        # Correction factor for the running stats.
        alpha = tl.math.exp2(m_i - m_ij)

        # Update the sum of attention weights.
        l_ij = tl.sum(p, 1)

        # Update the output accumulator.
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v_block.dtype), v_block, acc)

        # Update the running log-sum-exp and max.
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Advance pointers for the next iteration.
        k_ptrs += BLOCK_N * HEAD_DIM
        v_ptrs += BLOCK_N * HEAD_DIM

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=4),
    ],
    key=["N_CTX", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    M,
    stride_z,
    stride_h,
    stride_m,
    stride_k,
    sm_scale,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Main kernel for the forward pass of attention.

    Iterates over blocks of the query matrix and computes the corresponding
    output blocks.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Offsets for batch and head dimensions.
    q_offset = off_hz * stride_h
    k_offset = off_hz * stride_h
    o_offset = off_hz * stride_h
    v_offset = off_hz * stride_h

    # Pointers to Q, K, V, and O tensors.
    Q_block_ptr = (
        Q
        + q_offset
        + (
            start_m * BLOCK_M * stride_m
            + tl.arange(0, BLOCK_M)[:, None] * stride_m
            + tl.arange(0, HEAD_DIM)[None, :]
        )
    )
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    O_block_ptr = (
        O
        + o_offset
        + (
            start_m * BLOCK_M * stride_m
            + tl.arange(0, BLOCK_M)[:, None] * stride_m
            + tl.arange(0, HEAD_DIM)[None, :]
        )
    )

    # Pointers to the statistics tensor M (for numerical stability).
    M_ptr = M + off_hz * N_CTX

    # Initialize statistics and accumulator.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load the Q block, which will be reused.
    q = tl.load(Q_block_ptr)

    # --- Causal Attention Computation ---
    # The computation is split into two stages:
    # 1. Blocks before the diagonal.
    # 2. The diagonal block itself.

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Stage 1: Off-diagonal blocks (pre-causal).
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_ptr,
        V_ptr,
        sm_scale,
        start_m,
        N_CTX,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        1,
        offs_m,
        offs_n,
    )

    # Stage 2: Diagonal block (causal masking).
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_ptr,
        V_ptr,
        sm_scale,
        start_m,
        N_CTX,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        2,
        offs_m,
        offs_n,
    )

    # -- Finalize and Store Output --
    # Normalize the accumulator.
    acc = acc / l_i[:, None]

    # Store the output block.
    tl.store(O_block_ptr, acc.to(Q.dtype.element_ty))

    # Store the log-sum-exp for numerical stability in the backward pass (if implemented).
    m_i += tl.math.log2(l_i)
    tl.store(M_ptr + offs_m, m_i)


def attention_forward(q, k, v, sm_scale):
    """
    Python wrapper for the simplified Flash Attention forward pass.

    Args:
        q (torch.Tensor): Query tensor of shape (Z, H, N_CTX, HEAD_DIM).
        k (torch.Tensor): Key tensor of shape (Z, H, N_CTX, HEAD_DIM).
        v (torch.Tensor): Value tensor of shape (Z, H, N_CTX, HEAD_DIM).
        sm_scale (float): Scaling factor for the softmax.

    Returns:
        torch.Tensor: Output tensor of shape (Z, H, N_CTX, HEAD_DIM).
    """
    # Shape checks
    assert q.shape == k.shape == v.shape
    B, N_h, S, HEAD_DIM = q.shape

    assert HEAD_DIM in {16, 32, 64, 128, 256}

    # Output tensor
    o = torch.empty_like(q)
    # Temporary tensor for statistics (log-sum-exp)
    M = torch.empty((B, N_h, S), device=q.device, dtype=torch.float32)

    # Grid definition for Triton kernel launch
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * N_h)

    # Launch the kernel
    _attn_fwd[grid](
        q,
        k,
        v,
        o,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        sm_scale,
        N_CTX=S,
        HEAD_DIM=HEAD_DIM,
    )

    return o


if __name__ == "__main__":
    # Define tensor dimensions
    B, H, N_CTX, HEAD_DIM = 2, 4, 1024, 64
    sm_scale = 1.0 / (HEAD_DIM**0.5)

    # Create random input tensors on the GPU
    q = torch.randn((B, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
    k = torch.randn((B, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
    v = torch.randn((B, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")

    # --- Triton Implementation ---
    triton_output = attention_forward(q, k, v, sm_scale)

    # --- PyTorch Reference Implementation ---
    # For verification purposes
    q_ref, k_ref, v_ref = q.float(), k.float(), v.float()

    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q_ref, k_ref.transpose(2, 3)) * sm_scale
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p, dim=-1)
    ref_output = torch.matmul(p, v_ref).half()

    # --- Compare Results ---
    print("Triton output checksum:", triton_output.sum())
    print("PyTorch output checksum:", ref_output.sum())

    # Using torch.allclose for a more robust comparison
    are_close = torch.allclose(triton_output, ref_output, atol=1e-2, rtol=0)
    print(f"\nOutputs are close: {are_close}")

    # For more detailed comparison, check the mean absolute error
    mae = (triton_output - ref_output).abs().mean()
    print(f"Mean Absolute Error: {mae.item():.6f}")
