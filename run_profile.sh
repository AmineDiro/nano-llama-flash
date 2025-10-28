#!/bin/bash


# Profile triton kernel
TRITON_INTERPRET=0 TRITON_DEBUG=1 nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    -d 1 \
    --force-overwrite true \
    -o timeline \
    uv run python kernels/triton_flash_att_v2.py

# ncu CUDA flash attention
# NOTE: had to install ninja system global
sudo CUDA_HOME=/opt/cuda /usr/local/NVIDIA-Nsight-Compute/ncu \
    --set full \
    --kernel-name "flash_attention_kernel" \
    --import-source yes \
    -o profile_flashattn_cuda_only \
    -f .venv/bin/python3 kernels/flash_attention_cuda.py
