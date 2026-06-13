# RLLM_onnx_win

**Rust-native LLM inference engine for weak GPUs on Windows.**

Based on vLLM architecture, rewritten in Rust for maximum performance on budget hardware.
Runs local LLM models on NVIDIA T600, GTX 1050, and other low-end GPUs with CUDA 11.8+.

## What is this?

RLLM_onnx_win is a lightweight LLM inference engine designed for:
- **Weak GPUs** (2-4 GB VRAM): T600, GTX 1050, MX150, etc.
- **Windows** native (no Docker, no WSL)
- **CUDA 11.8** for maximum compatibility
- **Rust** for speed — the engine core is written in Rust via PyO3

## Features

| Feature | Status | Source |
|---------|--------|--------|
| PagedAttention | Working | `attention/paged.rs` — block-based KV cache management |
| CUDA 11.8 support | Working | cudarc 0.12 + E:\CUDA\v11.8 |
| GPU detection | Working | `rllm.cuda_info()` — NVIDIA T600 CC 75 confirmed |
| FP8 quantization | Implemented | `quantization/fp8.rs` |
| INT8 quantization | Implemented | `quantization/int8.rs` |
| INT4 quantization | Implemented | `quantization/int4.rs` |
| CPU backend | Implemented | `cpu/mod.rs` — fallback for no-GPU systems |
| MoE routing | Implemented | `moe/mod.rs` — Mixture of Experts |
| HTTP server | Implemented | `server/mod.rs` — axum-based, OpenAI API |
| Tokenizer | Implemented | `tokenizer/mod.rs` — HuggingFace tokenizers |
| PyO3 Python bindings | Working | `import rllm` in Python 3.10 |
| Session recovery | Working | browser-server.js — no crash dialogs |

## Quick Start

### Build
```bash
set PYO3_PYTHON=E:\Python310\python.exe
cd RLLM_onnx_win
cargo build --release
```

### Chat with model
```bash
chat.bat
```

### Python API
```python
import rllm

# GPU info
print(rllm.cuda_info())
# → Devices: 1, Name: NVIDIA T600, Compute Capability: 75

# PagedAttention
pa = rllm.PyPagedAttention(num_blocks=128, block_size=16, num_layers=32, num_heads=32, head_dim=128)
pa.allocate_seq(1)  # allocate KV block for sequence 1
print(pa.stats())
# → Blocks: 127/128 free, block_size=16, kv_per_block=524288 bytes
```

## Architecture

```
RLLM_onnx_win/
├── src/
│   ├── lib.rs              — PyO3 module entry point
│   ├── core/
│   │   ├── cumem_allocator.rs  — CUDA memory pool (port of vLLM csrc/)
│   │   └── registration.rs     — PyTorch op registration
│   ├── cuda/mod.rs          — CUDA device management (cudarc 0.12)
│   ├── attention/
│   │   ├── mod.rs           — Flash/Paged Attention backend
│   │   └── paged.rs         — Block table & KV cache management
│   ├── quantization/
│   │   ├── fp8.rs           — FP8 per-token-group quantization
│   │   ├── int8.rs          — INT8 quantization
│   │   └── int4.rs          — INT4 packed quantization
│   ├── moe/mod.rs           — Mixture of Experts routing
│   ├── cpu/mod.rs           — CPU fallback backend
│   ├── server/mod.rs        — HTTP/gRPC server (axum)
│   └── tokenizer/mod.rs     — HuggingFace tokenizer wrapper
├── Cargo.toml               — Rust dependencies
├── build.rs                 — CUDA compilation via nvcc
├── build.bat                — Build & deploy to Python
└── chat.bat                 — Interactive model chat
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Engine core | **Rust** (edition 2024) | Memory safety + zero-cost abstractions |
| CUDA bindings | **cudarc 0.12** | Safe Rust CUDA driver API |
| Python bindings | **PyO3 0.23** | Seamless Python ↔ Rust interop |
| HTTP server | **axum 0.8** | High-performance async web framework |
| Tokenizer | **tokenizers 0.22** | HuggingFace fast tokenizer |
| Quantization | **half + ndarray** | Efficient low-precision math |

## Hardware Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+ (`E:\Python310`)
- **PyTorch**: 2.7.1+cu118
- **CUDA**: 11.8+ (`E:\CUDA\v11.8`)
- **GPU**: NVIDIA with CUDA support (tested on T600 4GB)
- **Rust**: 1.95+ (`rustup`)

## Based on

- [vLLM](https://github.com/vllm-project/vllm) — original LLM serving engine (Python/C++/CUDA)
- [cudarc](https://github.com/coreylowman/cudarc) — safe CUDA bindings for Rust
- [uv](https://github.com/astral-sh/uv) — inspired by Rust approach to package management speed

## Performance (T600, SmolLM 135M)

| Metric | Value |
|--------|-------|
| VRAM usage | 256 MB |
| Inference speed | 8-9 tok/s |
| PagedAttention blocks | 128 × 16 tokens |
| KV cache per block | 512 KB |
| Total KV cache | 64 MB |

## License

Apache-2.0 / MIT
