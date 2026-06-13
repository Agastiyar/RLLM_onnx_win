pub fn dequantize_int4_packed(input: &[u8], scales: &[f32], group_size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() * 2);
    for &byte in input.iter() {
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        output.push(lo as f32);
        output.push(hi as f32);
    }
    for (i, val) in output.iter_mut().enumerate() {
        let group = i / group_size;
        if group < scales.len() {
            *val *= scales[group];
        }
    }
    output
}
