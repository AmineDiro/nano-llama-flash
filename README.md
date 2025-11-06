# llama-flash

A from-scratch implementation of state-of-the-art transformer components for deep understanding. This project explores the internals of modern LLMs by rebuilding key components from the ground up.

## What's Implemented

### Core Architecture (`model.py`)
- **Llama-style Transformer**: Complete decoder-only architecture with:
  - Multi-head attention with Grouped Query Attention (GQA) support
  - KV caching for efficient inference
  - RMSNorm for layer normalization
  - SwiGLU-style MLP with GELU activation

- **RoPE (Rotary Position Embeddings)**: Full implementation of positional encoding through rotation in complex space

### Flash Attention Kernels (`kernels/`)
Multiple implementations to understand performance characteristics and limitations:
- **CUDA Kernels** (`flash_attention_kernel.cu`): Custom CUDA implementation
- **Triton Kernels**: Various versions exploring different optimizations
  - Base implementation (`triton_flash_att.py`)
  - V2 with improvements (`triton_flash_att_v2.py`)
  - Padded sequence handling (`triton_flash_att_v2_padded.py`)
  - Reference forward pass (`triton_ref_impl_fwd.py`)

## TODO
- [ ] GRPO (Group Relative Policy Optimization) implementation

## Purpose

This project is an educational deep-dive into modern transformer architectures. By implementing these components from scratch, I'm building intuition for how they work under the hood and understanding their performance characteristics.

Blog post coming soon.
