#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/types.h>
#include <cuda.h>
#include <math.h>

// Flash Attention CUDA Kernel - Direct translation from Triton
// No optimizations - straightforward implementation for readability

__global__ void flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int S,           // Sequence length
    int stride_H,    // Stride between heads
    int D,           // Head dimension
    float softmax_scale,
    int Tc,          // Number of K/V tiles (S / Bc)
    int Bc           // Block size
) {
    int pid_x = blockIdx.x;  // Q block index
    int pid_y = blockIdx.y;  // batch * head index
    int tid = threadIdx.x;   // Thread index within block

    // Offset to current batch/head
    int batch_offset = pid_y * stride_H;
    const float* q_ptr = Q + batch_offset;
    const float* k_ptr = K + batch_offset;
    const float* v_ptr = V + batch_offset;
    float* o_ptr = O + batch_offset;

    // Shared memory for tiles
    extern __shared__ float smem[];
    float* qi = smem;                    // [Bc][D]
    float* kj = smem + Bc * D;           // [Bc][D]
    float* vj = smem + 2 * Bc * D;       // [Bc][D]
    float* Sij = smem + 3 * Bc * D;      // [Bc][Bc]
    float* prev_mi = smem + 3 * Bc * D + Bc * Bc;  // [Bc]
    float* prev_li = smem + 3 * Bc * D + Bc * Bc + Bc;  // [Bc]
    float* acc = smem + 3 * Bc * D + Bc * Bc + 2 * Bc;  // [Bc][D]
    float* mij = smem + 3 * Bc * D + Bc * Bc + 2 * Bc + Bc * D;  // [Bc]
    float* lij = smem + 3 * Bc * D + Bc * Bc + 2 * Bc + Bc * D + Bc;  // [Bc]

    // Initialize running statistics cooperatively
    for (int i = tid; i < Bc; i += blockDim.x) {
        prev_mi[i] = -INFINITY;
        prev_li[i] = 0.0f;
    }
    for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
        acc[idx] = 0.0f;
    }
    __syncthreads();

    // Load Query block Q_i: shape (Bc, D)
    // Each thread loads multiple elements cooperatively
    int q_row_start = pid_x * Bc;
    for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
        int row = idx / D;
        int col = idx % D;
        int global_row = q_row_start + row;

        if (global_row < S) {
            qi[row * D + col] = q_ptr[global_row * D + col];
        } else {
            qi[row * D + col] = 0.0f;
        }
    }
    __syncthreads();

    // Loop over K/V blocks (inner loop)
    for (int j = 0; j < Tc; j++) {
        // Load K_j block: shape (Bc, D)
        int k_row_start = j * Bc;
        for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            int global_row = k_row_start + row;

            if (global_row < S) {
                kj[row * D + col] = k_ptr[global_row * D + col];
            } else {
                kj[row * D + col] = 0.0f;
            }
        }

        // Load V_j block: shape (Bc, D)
        for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
            int row = idx / D;
            int col = idx % D;
            int global_row = k_row_start + row;

            if (global_row < S) {
                vj[row * D + col] = v_ptr[global_row * D + col];
            } else {
                vj[row * D + col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores: Sij = Q_i @ K_j^T * softmax_scale
        // Shape: (Bc, Bc)
        // Simple matrix multiplication - no optimization
        for (int idx = tid; idx < Bc * Bc; idx += blockDim.x) {
            int row = idx / Bc;  // Q row
            int col = idx % Bc;  // K row

            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += qi[row * D + d] * kj[col * D + d];  // K transposed
            }
            Sij[row * Bc + col] = sum * softmax_scale;
        }
        __syncthreads();

        // Compute row-wise max and softmax statistics
        // Each thread handles one or more rows
        // TODO: Optimize with warp shuffle intrinsics later
        for (int row = tid; row < Bc; row += blockDim.x) {
            // Row-wise max
            float row_max = -INFINITY;
            for (int col = 0; col < Bc; col++) {
                row_max = fmaxf(row_max, Sij[row * Bc + col]);
            }
            mij[row] = row_max;

            // Compute exp and row-wise sum
            float row_sum = 0.0f;
            for (int col = 0; col < Bc; col++) {
                float pij_val = expf(Sij[row * Bc + col] - row_max);
                Sij[row * Bc + col] = pij_val;  // Store P_ij back in Sij
                row_sum += pij_val;
            }
            lij[row] = row_sum;
        }
        __syncthreads();

        // Update running statistics and accumulator
        for (int row = tid; row < Bc; row += blockDim.x) {
            // New maximum
            float mi_new = fmaxf(prev_mi[row], mij[row]);

            // Scaling factors
            float alpha = expf(prev_mi[row] - mi_new);
            float beta = expf(mij[row] - mi_new);

            // Update running sum
            float li_new = prev_li[row] * alpha + lij[row] * beta;

            // Update accumulator: acc = alpha * acc + beta * P_ij @ V_j
            // First scale existing accumulator
            for (int d = 0; d < D; d++) {
                acc[row * D + d] *= alpha;
            }
            // Add beta * P_ij @ V_j

            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int col = 0; col < Bc; col++) {
                    sum += Sij[row * Bc + col] * vj[col * D + d];
                }
                acc[row * D + d] += beta * sum;
            }

            // Update running statistics
            prev_mi[row] = mi_new;
            prev_li[row] = li_new;
        }
        __syncthreads();
    }

    // Normalize and store output
    for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
        int row = idx / D;
        int col = idx % D;
        int global_row = q_row_start + row;

        if (global_row < S) {
            float normalized = acc[row * D + col] / prev_li[row];
            o_ptr[global_row * D + col] = normalized;
        }
    }

    __syncthreads();
}

torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const int Bc
) {
    const int B = Q.size(0);
    const int N_h = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    int Tc = (S + Bc - 1) / Bc;  // Number of tiles == Tr

    int stride_H = Q.stride(1);
    float softmax_scale = 1.0f / sqrtf((float)D);
    dim3 grid((S + Bc - 1) / Bc, B * N_h, 1);

    auto O = torch::zeros_like(Q);

    // TODO: maybe use more warps, this uses 1 warp(32)
    // Block: Bc threads per block
    dim3 block(Bc, 1, 1);

    // Shared memory size: 3 * Bc * D (for qi, kj, vj) + Bc * Bc (for Sij)
    //                   + Bc (prev_mi) + Bc (prev_li) + Bc * D (acc) + Bc (mij) + Bc (lij)
    size_t smem_size = (4 * Bc * D + Bc * Bc + 4 * Bc) * sizeof(float);

    flash_attention_kernel<<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(), 
        O.data_ptr<float>(),
        S,
        stride_H, 
        D, 
        softmax_scale,
        Tc, 
        Bc
    );


    return O;

}
