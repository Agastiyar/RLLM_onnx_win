//! PagedAttention — port of csrc/attention/
//!
//! Memory-efficient attention with paged KV cache.
//! Block-based management of attention key/value tensors.

use std::collections::VecDeque;

/// A single block in the KV cache (like a page in virtual memory).
#[derive(Debug, Clone)]
pub struct KVBlock {
    pub block_id: usize,
    pub seq_ids: Vec<u64>,
    pub ref_count: usize,
}

/// Block table mapping logical blocks to physical blocks.
#[derive(Debug)]
pub struct BlockTable {
    block_size: usize,
    num_blocks: usize,
    free_blocks: VecDeque<usize>,
    blocks: Vec<Option<KVBlock>>,
    block_to_seq: Vec<Vec<u64>>,
}

impl BlockTable {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free: VecDeque<usize> = (0..num_blocks).collect();
        Self {
            block_size,
            num_blocks,
            free_blocks: free,
            blocks: vec![None; num_blocks],
            block_to_seq: vec![Vec::new(); num_blocks],
        }
    }

    pub fn allocate(&mut self, seq_id: u64) -> Option<usize> {
        let block_id = self.free_blocks.pop_front()?;
        self.blocks[block_id] = Some(KVBlock {
            block_id,
            seq_ids: vec![seq_id],
            ref_count: 1,
        });
        self.block_to_seq[block_id] = vec![seq_id];
        Some(block_id)
    }

    pub fn free(&mut self, block_id: usize) {
        self.blocks[block_id] = None;
        self.block_to_seq[block_id].clear();
        self.free_blocks.push_back(block_id);
    }

    pub fn free_all_for_seq(&mut self, seq_id: u64) -> usize {
        let mut freed = 0;
        for i in 0..self.num_blocks {
            if self.block_to_seq[i].contains(&seq_id) {
                self.free(i);
                freed += 1;
            }
        }
        freed
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }
}

/// PagedAttention engine managing KV cache blocks for multiple sequences.
pub struct PagedAttentionEngine {
    block_table: BlockTable,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl PagedAttentionEngine {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            block_table: BlockTable::new(num_blocks, block_size),
            num_layers,
            num_heads,
            head_dim,
        }
    }

    pub fn allocate_seq(&mut self, seq_id: u64) -> Option<usize> {
        self.block_table.allocate(seq_id)
    }

    pub fn free_seq(&mut self, seq_id: u64) -> usize {
        self.block_table.free_all_for_seq(seq_id)
    }

    pub fn seq_num_blocks(&self, seq_id: u64) -> usize {
        self.block_table.block_to_seq.iter()
            .filter(|blocks| blocks.contains(&seq_id))
            .count()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_table.num_free()
    }

    pub fn num_used_blocks(&self) -> usize {
        self.block_table.num_used()
    }

    pub fn kv_cache_bytes_per_block(&self) -> usize {
        2 * self.num_layers * self.num_heads * self.head_dim * 2
    }

    pub fn stats(&self) -> BlockTableStats {
        BlockTableStats {
            total_blocks: self.block_table.num_blocks,
            free_blocks: self.block_table.num_free(),
            used_blocks: self.block_table.num_used(),
            block_size: self.block_table.block_size(),
            kv_bytes_per_block: self.kv_cache_bytes_per_block(),
        }
    }
}

#[derive(Debug)]
pub struct BlockTableStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub block_size: usize,
    pub kv_bytes_per_block: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_alloc_free() {
        let mut bt = BlockTable::new(4, 16);
        assert_eq!(bt.num_free(), 4);

        let b0 = bt.allocate(1).unwrap();
        let b1 = bt.allocate(1).unwrap();
        let b2 = bt.allocate(2).unwrap();
        assert_eq!(bt.num_free(), 1);
        assert_eq!(bt.num_used(), 3);

        bt.free(b1);
        assert_eq!(bt.num_free(), 2);

        let freed = bt.free_all_for_seq(1);
        assert_eq!(freed, 1);
        assert_eq!(bt.num_free(), 3);
    }
}
