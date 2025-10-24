# Implementing general transformer kernels for learning stuff

- [x] Implement transformer archi
- [x] Implement simple `flash_attention` kernel in triton
- [ ] Implement a generate + sampling
- [ ] Implement GRPO simple

- [ ] Run a chat template
# Later

---

### FlashAttention 
#### TODO List for V1 Polish

- [ ] **Backward Pass Kernel:** If you were to implement the backward pass, the TODO would be: "Create a corresponding backward kernel that recomputes `Sij` blocks on-the-fly using the saved `O`, `l_i`, and `m_i` statistics."


#### TODO List for V2 Upgrade
- [ ] **Refactor Kernel Launch Grid:** Change the kernel launch from `(num_blocks_c, B*H)` to `(num_blocks_c, num_blocks_r, B*H)`. Each `(pid_x, pid_y)` now corresponds to a single `Sij` block computation.
- [ ] **Eliminate the Inner Loop:** Remove the `for j in range(0, Tc):` loop. The parallelism over `j` is now handled by launching more thread blocks (`pid_y`).
- [ ] **Manage Partial Outputs:**
    -   Each thread block will now compute a partial accumulator, a local max `mi`, and a local sum `li`.
    -   These partial results must be written back to HBM (or a temporary workspace).
- [ ] **Implement a Reduction Kernel:** Create a second, simple kernel that loads all the partial results for a given query block, finds the true global maximum, correctly rescales the partial accumulators, sums them up, and writes the final, normalized output.



#### TODO List for V3 Upgrade: FlashAttention v3: Hardware-Aware Asynchrony (Hopper GPU)
- [ ] **Implement Asynchronous Loads (Pipelining):**
    -   Restructure the kernel to create a software pipeline. In the main loop, issue an async load for iteration `j+1` (`tl.copy_async`) *before* starting the computation for iteration `j`.
    -   Use a synchronization primitive (`tl.cp_async_wait()`) to ensure the data for iteration `j` has arrived before it is used.
- [ ] **Utilize Hardware-Specific Matmul Instructions:** Ensure your `tl.dot` call is configured with block sizes and data types that allow Triton to compile it down to Hopper's powerful WGMMA instructions.
- [ ] **Add FP8 Data Handling:**
    -   Modify the kernel signature to accept pointers to FP8 tensors (`tl.float8e4nv` or similar).
    -   Introduce logic to handle the scaling factors associated with FP8, as computation is often done in a higher precision format like FP16 or FP32.

#### FA4: Blackwell GPUs
1.  **Smarter Softmax Stability:** The algorithm is refined to perform the expensive rescaling of the accumulator only when the running max `mi` changes by a numerically significant amount, avoiding unnecessary computation.
2.  **Avoiding Special Function Units (SFUs):** The `exp()` function is often a bottleneck as it relies on limited SFUs. V4 may use a faster, software-based polynomial approximation of the exponential function, computed directly on the CUDA cores.
3.  **Deeper Warp Specialization:** The division of labor is even finer. There might be dedicated warps for loading, MMAs, softmax calculations, and numerical correction steps, creating a highly optimized, multi-stage pipeline within a single thread block.
- [ ] **Implement Conditional Rescaling:** In the online softmax logic, add a condition: `if mij > prev_mi:`. The expensive rescaling of `acc` and `li` should only happen inside this block. If the max doesn't change, the update is much simpler.

#### TODO List for V4 Upgrade
- [ ] **Replace `tl.exp` with a Polynomial Approximation:** This is a highly advanced step.
    -   Find or derive a polynomial that approximates `e^x` accurately enough within the expected range of values.
    -   Replace the `tl.exp(Sij - mij[:, None])` call with your custom implementation using basic arithmetic operations (`*`, `+`).
- [ ] **Advanced Software Pipelining:** Design a state machine for warps within the thread block. A warp's role (loader, computer, updater) could change as the kernel progresses, requiring complex shared memory management and synchronization (`tl.barrier`) to ensure correctness.
