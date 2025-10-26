#!/bin/bash


TRITON_INTERPRET=0 TRITON_DEBUG=1 nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    -d 1 \
    --force-overwrite true \
    -o timeline \
    uv run python kernels/triton_flash_att_v2.py
