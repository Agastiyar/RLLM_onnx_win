//! CPU kernels — port of csrc/cpu/

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }

    pub fn gemm_f32(
        &self,
        a: &[f32], m: usize, k: usize,
        b: &[f32], n: usize,
        c: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
