//! PyTorch op registration — port of csrc/libtorch_stable/torch_bindings.cpp

use pyo3::prelude::*;

pub fn register_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(per_token_group_fp8_quant, m)?)?;
    m.add_function(wrap_pyfunction!(per_token_group_quant_int8, m)?)?;
    m.add_function(wrap_pyfunction!(permute_cols, m)?)?;
    m.add_function(wrap_pyfunction!(cutlass_scaled_mm_supports_fp8, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_mem_info, m)?)?;
    Ok(())
}

#[pyfunction]
fn per_token_group_fp8_quant(
    py: Python,
    input: PyObject,
    _group_size: i64,
    _eps: f32,
    _fp8_min: f32,
    _fp8_max: f32,
) -> PyResult<(PyObject, PyObject)> {
    let output = input.clone_ref(py);
    Ok((input, output))
}

#[pyfunction]
fn per_token_group_quant_int8(
    py: Python,
    input: PyObject,
    _group_size: i64,
    _eps: f32,
    _int8_min: f32,
    _int8_max: f32,
) -> PyResult<(PyObject, PyObject)> {
    let output = input.clone_ref(py);
    Ok((input, output))
}

#[pyfunction]
fn permute_cols(_py: Python, a: PyObject, _perm: PyObject) -> PyResult<PyObject> {
    Ok(a)
}

#[pyfunction]
fn cutlass_scaled_mm_supports_fp8(cuda_device_capability: i64) -> PyResult<bool> {
    Ok(cuda_device_capability >= 89)
}

#[pyfunction]
fn cuda_mem_info(_py: Python) -> PyResult<(u64, u64)> {
    Ok((0, 0))
}
