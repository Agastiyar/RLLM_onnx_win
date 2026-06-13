//! Quantization — port of csrc/quantization/
//!
//! FP8, INT8, INT4 quantization kernels.
//! CUDA kernels stay in .cu, Rust manages the dispatch.

pub mod fp8;
pub mod int8;
pub mod int4;
