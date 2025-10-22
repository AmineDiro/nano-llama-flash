import triton
import triton.language as tl


@triton.jit
def add_kernel(
    A,  # pointer to a
    B,  # pointer to b
    C,  # pointer to output
    N,  # pointer to the nb of elements
    stride_A,
    stride_B,
    stride_C,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    # Get the thread index
    pid = tl.program_id(0)
    # Create a local index for each element
    # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    # This like pytorch -> adding a float to a vec
    idx = block_start + tl.arange(0, BLOCK_SIZE)

    # Ensure we do not go out of bounds
    mask = idx < N

    # Load data from A and B
    a = tl.load(A + idx * stride_A, mask=mask)
    b = tl.load(B + idx * stride_B, mask=mask)

    # Add the two arrays
    c = a + b

    # Store the result in C
    tl.store(C + idx * stride_C, c, mask=mask)
