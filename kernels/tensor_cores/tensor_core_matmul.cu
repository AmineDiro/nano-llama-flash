#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// Define the tile dimensions for the Tensor Core unit
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel: C = A * B + C
// A and B are FP16 (__half), C and D are FP32 (float)
// This implements a naive tiling strategy where one Warp computes one 16x16 output tile.
__global__ void wmma_ker(half *a, half *b, float *c, int M, int N, int K) {
    
    // 1. Calculate the global warp ID
    // threadIdx.x contains the thread ID within the block
    // blockIdx.x contains the block ID
    // warpSize is usually 32
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    // 2. Declare the fragments
    // These behave like registers but hold a matrix tile
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 3. Initialize the output accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // 4. Loop over the K-dimension
    // We move across the A rows and B columns in chunks of 16 (WMMA_K)
    for (int i = 0; i < K; i += WMMA_K) {
        
        // Calculate pointers for the current tile in A and B
        // aRow = warpM * WMMA_M (Starting row for this warp)
        // aCol = i (Current K step)
        int aRow = warpM * WMMA_M;
        int aCol = i;
        
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Check bounds (basic safety)
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            
            // Load the inputs
            // The last argument is the "Leading Dimension" (stride)
            // For Row Major A, stride is K. For Row Major B, stride is N.
            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // Perform the matrix multiplication
            // c_frag = a_frag * b_frag + c_frag
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // 5. Store the result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        // Store the FP32 accumulator back to global memory
        // Stride is N
        wmma::store_matrix_sync(c + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Host helper to run the kernel
void run_matmul() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    printf("Initializing matrices %dx%d...\n", M, N);

    // Host memory
    half *h_a = new half[M * K];
    half *h_b = new half[K * N];
    float *h_c = new float[M * N];

    // Initialize with dummy data
    for(int i=0; i < M*K; i++) h_a[i] = __float2half(1.0f);
    for(int i=0; i < K*N; i++) h_b[i] = __float2half(1.0f);

    // Device memory
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Kernel Configuration
    // Each warp calculates a 16x16 tile.
    // We need M/16 warps in Y direction and N/16 warps in X direction (conceptually)
    
    int warps_per_block = 4; // 128 threads
    dim3 gridDim( (M + (WMMA_M * warps_per_block - 1)) / (WMMA_M * warps_per_block), 
                  (N + WMMA_N - 1) / WMMA_N );
    dim3 blockDim(32 * warps_per_block);

    printf("Launching Kernel...\n");
    wmma_ker<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    printf("Done. Checking first element...\n");
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Result should be 1.0 * 1.0 * 1024 = 1024.0
    printf("Result[0] = %f (Expected 1024.0)\n", h_c[0]);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}

int main() {
    run_matmul();
    return 0;
}
