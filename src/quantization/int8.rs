pub fn dequantize_int8(input: &[i8], scales: &[f32], group_size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for (i, &val) in input.iter().enumerate() {
        let group = i / group_size;
        output[i] = val as f32 * scales[group];
    }
    output
}
