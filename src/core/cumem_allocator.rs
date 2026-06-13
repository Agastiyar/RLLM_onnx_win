//! CUDA memory allocator — port of csrc/cumem_allocator.cpp

use parking_lot::Mutex;

pub struct CuMemBlock {
    pub ptr: u64,
    pub size: u64,
}

pub struct CuMemAllocator {
    block_size: u64,
    total: Mutex<u64>,
}

impl CuMemAllocator {
    pub fn new(block_size: u64) -> Self {
        Self {
            block_size,
            total: Mutex::new(0),
        }
    }

    pub fn allocate(&self, size: u64) -> CuMemBlock {
        let aligned = (size + self.block_size - 1) & !(self.block_size - 1);
        *self.total.lock() += aligned;
        CuMemBlock { ptr: 0, size: aligned }
    }

    pub fn total_allocated(&self) -> u64 {
        *self.total.lock()
    }
}
