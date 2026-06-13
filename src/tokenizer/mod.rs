//! Tokenizer — port of rust/src/tokenizer/

pub struct RllmTokenizer {
    inner: tokenizers::Tokenizer,
}

impl RllmTokenizer {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Tokenizer error: {e}"))?;
        Ok(Self { inner: tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| format!("Encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, false)
            .map_err(|e| format!("Decode error: {e}"))
    }
}
