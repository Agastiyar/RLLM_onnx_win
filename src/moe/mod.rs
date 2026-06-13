//! MoE (Mixture of Experts) — port of csrc/moe/

pub struct MoERouter {
    num_experts: usize,
    top_k: usize,
}

impl MoERouter {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    pub fn route(&self, gating_output: &[f32]) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = gating_output.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(self.top_k);
        scores
    }
}
