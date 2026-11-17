// SPDX-License-Identifier: Apache-2.0
//! The fixtures' random vectors (`tests/scripts/gen_vector_goldens.py`), shared by the vector golden tests.

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The fixture's random vector: `float32((unit - 0.5) * 2 * scale)`, the stream seeded `seed * 2 + role`.
pub(crate) fn random_vector(dim: usize, seed: u64, role: u64, scale: f64) -> Vec<Option<f32>> {
    let mut state = seed * 2 + role;
    (0..dim)
        .map(|_| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the 53-bit unit of the generator"
            )]
            let unit = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "float32 rounding is the point"
            )]
            let v = ((unit - 0.5) * 2.0 * scale) as f32;
            Some(v)
        })
        .collect()
}
