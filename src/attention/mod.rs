//! Attention kernels — port of csrc/attention/
pub mod paged;

use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct AttentionBackend {
    capability: i32,
}

impl AttentionBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let capability = device.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0) * 10
            + device.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
        Self { capability }
    }

    pub fn supports_flash_attention(&self) -> bool {
        self.capability >= 80
    }

    pub fn supports_paged_attention(&self) -> bool {
        self.capability >= 70
    }

    pub fn capability(&self) -> i32 {
        self.capability
    }
}

pub use paged::{PagedAttentionEngine, BlockTableStats};
