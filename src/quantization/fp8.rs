use half::f16;

#[derive(Debug, Clone, Copy)]
pub enum QuantDtype {
    Fp8,
    Int8,
    Int4,
    Nf4,
}

pub fn per_token_group_fp8_quant(
    input: &[f16],
    group_size: usize,
    eps: f32,
    fp8_min: f32,
    fp8_max: f32,
) -> (Vec<u8>, Vec<f32>) {
    let n = input.len();
    let n_groups = n / group_size;
    let mut output_q = vec![0u8; n];
    let mut output_s = vec![0.0f32; n_groups];

    for g in 0..n_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group_slice = &input[start..end];

        let amax = group_slice.iter()
            .map(|x| x.to_f32().abs())
            .fold(0.0f32, f32::max);

        let scale = if amax > eps { amax / fp8_max } else { 1.0 };

        for (i, &val) in group_slice.iter().enumerate() {
            let v = (val.to_f32() / scale).clamp(fp8_min, fp8_max);
            output_q[start + i] = v.round() as u8;
        }
        output_s[g] = scale;
    }

    (output_q, output_s)
}

pub fn per_token_group_int8_quant(
    input: &[f16],
    group_size: usize,
    eps: f32,
    int8_min: f32,
    int8_max: f32,
) -> (Vec<i8>, Vec<f32>) {
    let n = input.len();
    let n_groups = n / group_size;
    let mut output_q = vec![0i8; n];
    let mut output_s = vec![0.0f32; n_groups];

    for g in 0..n_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group_slice = &input[start..end];

        let amax = group_slice.iter()
            .map(|x| x.to_f32().abs())
            .fold(0.0f32, f32::max);

        let scale = if amax > eps { amax / int8_max } else { 1.0 };

        for (i, &val) in group_slice.iter().enumerate() {
            let v = (val.to_f32() / scale).clamp(int8_min, int8_max);
            output_q[start + i] = v.round() as i8;
        }
        output_s[g] = scale;
    }

    (output_q, output_s)
}
