pub mod core;
pub mod cuda;
pub mod cpu;
pub mod attention;
pub mod quantization;
pub mod moe;
pub mod server;
pub mod tokenizer;

use pyo3::prelude::*;
use std::sync::Mutex;

#[pymodule]
fn rllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::registration::register_ops(m)?;
    m.add_class::<RllmConfig>()?;
    m.add_class::<PyPagedAttention>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_info, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn cuda_info() -> PyResult<String> {
    match cuda::init_device() {
        Ok(device) => {
            let name = cuda::get_device_name(&device);
            let cc = cuda::get_compute_capability(&device);
            let count = cuda::get_device_count();
            Ok(format!("Devices: {count}, Name: {name}, Compute Capability: {cc}"))
        }
        Err(e) => Ok(format!("CUDA not available: {e}")),
    }
}

#[pyclass]
#[derive(Clone, Debug)]
struct RllmConfig {
    #[pyo3(get, set)]
    pub model_path: String,
    #[pyo3(get, set)]
    pub tensor_parallel_size: u32,
    #[pyo3(get, set)]
    pub max_model_len: u32,
    #[pyo3(get, set)]
    pub gpu_memory_utilization: f32,
    #[pyo3(get, set)]
    pub quantization: String,
}

#[pymethods]
impl RllmConfig {
    #[new]
    #[pyo3(signature = (model_path, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9, quantization="none".to_string()))]
    fn new(
        model_path: String,
        tensor_parallel_size: u32,
        max_model_len: u32,
        gpu_memory_utilization: f32,
        quantization: String,
    ) -> Self {
        Self {
            model_path,
            tensor_parallel_size,
            max_model_len,
            gpu_memory_utilization,
            quantization,
        }
    }
}

#[pyclass]
struct PyPagedAttention {
    engine: Mutex<attention::PagedAttentionEngine>,
}

#[pymethods]
impl PyPagedAttention {
    #[new]
    #[pyo3(signature = (num_blocks=64, block_size=16, num_layers=32, num_heads=32, head_dim=128))]
    fn py_new(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            engine: Mutex::new(attention::PagedAttentionEngine::new(
                num_blocks, block_size, num_layers, num_heads, head_dim,
            )),
        }
    }

    fn allocate_seq(&self, seq_id: u64) -> Option<usize> {
        self.engine.lock().unwrap().allocate_seq(seq_id)
    }

    fn free_seq(&self, seq_id: u64) -> usize {
        self.engine.lock().unwrap().free_seq(seq_id)
    }

    fn seq_num_blocks(&self, seq_id: u64) -> usize {
        self.engine.lock().unwrap().seq_num_blocks(seq_id)
    }

    fn num_free_blocks(&self) -> usize {
        self.engine.lock().unwrap().num_free_blocks()
    }

    fn num_used_blocks(&self) -> usize {
        self.engine.lock().unwrap().num_used_blocks()
    }

    fn stats(&self) -> String {
        let s = self.engine.lock().unwrap().stats();
        format!(
            "Blocks: {}/{} free, block_size={}, kv_per_block={} bytes",
            s.free_blocks, s.total_blocks, s.block_size, s.kv_bytes_per_block
        )
    }
}
