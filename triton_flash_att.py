import math
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# softmax((Q.K^T)/sqrt(d)).V
@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    S,
    D,
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

    # offset the batch*N_h, for each dim, skip to the next dim
    for j in range(0, Tc):
        # Load K_j, V_j from HBM to SRAM
        kj = K + batch_offset + j * Bc
        for i in range(0, Tr):
            # Load Q_i, O_i, l_i, m_i from HBM to SRAM
            pass

    tl.device_print("pid_x", pid_x)
    tl.device_print("pid_y", pid_y)


def main():
    device_properties = torch.cuda.get_device_properties(0)

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

    # NOTE:
    # (3 * Br * D_h * sizeof(float)) -> tile of q,k,v
    # (Br * Bc * sizeof(float)) -> tile scores (Br,Bc)
    sram_needed = (3 * Br * D_h * 4) + (Bc * Br * 4)

    max_sram_size = device_properties.shared_memory_per_block
    print(f"Device Name: {device_properties.name}")
    print(f"Maximum Shared Memory (SRAM) Per Block: {max_sram_size} bytes")
    print(f"Shared Memory needed: {sram_needed} bytes")

    attn_kernel[(B, N_h)](q, v, k, o, S, D_h, Tc, Tr, Bc, Br, math.sqrt(D_h), l, m)
    # grid = lambda _:

    # Verify the result using PyTorch

    # Check if the results are the same
    # assert torch.allclose(C, C_pytorch, atol=1e-6)


main()
