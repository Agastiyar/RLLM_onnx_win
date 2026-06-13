//! CUDA utilities — port of csrc/cuda_utils.h
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

pub type Device = Arc<CudaDevice>;

pub fn init_device() -> Result<Device, String> {
    CudaDevice::new(0).map_err(|e| e.to_string())
}

pub fn get_device_count() -> u32 {
    CudaDevice::count().unwrap_or(0) as u32
}

pub fn get_device_name(device: &Device) -> String {
    device.name().unwrap_or_else(|_| "Unknown GPU".to_string())
}

pub fn get_compute_capability(device: &Device) -> i32 {
    use cudarc::driver::sys::CUdevice_attribute as A;
    device.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0) * 10
        + device.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0)
}

pub fn copy_host_to_device<T: cudarc::driver::DeviceRepr>(
    device: &Device,
    src: &[T],
) -> Result<CudaSlice<T>, String> {
    device.htod_sync_copy(src).map_err(|e| e.to_string())
}

pub fn copy_device_to_host<T: cudarc::driver::DeviceRepr + Clone>(
    device: &Device,
    src: &CudaSlice<T>,
) -> Result<Vec<T>, String> {
    device.dtoh_sync_copy(src).map_err(|e| e.to_string())
}
