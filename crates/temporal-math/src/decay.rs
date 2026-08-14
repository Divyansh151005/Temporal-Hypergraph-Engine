//! Continuous exponential time decay: `w(t) = w0 * exp(-lambda * dt)`, with
//! `lambda = ln(2) / half_life` and `dt` measured in days.

use std::f32::consts::LN_2;

use wide::{f32x8, i32x8};

/// Seconds in one day, used to convert epoch-second deltas into days.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

const LANES: usize = 8;

/// Timestamps are converted in stack blocks of this many elements so that
/// `batch_compute_decay` never allocates.
const CONVERSION_BLOCK: usize = 512;

/// `exp(x)` underflows to zero in `f32` below roughly -87.3; clamping keeps the
/// SIMD polynomial inside its accurate domain.
const MIN_EXPONENT: f32 = -87.0;

/// Parameters of the exponential decay curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayConfig {
    /// Age (in days) at which a fact retains half of its base weight.
    pub half_life_days: f32,
    /// Base extraction confidence `w0`.
    pub base_weight: f32,
}

impl DecayConfig {
    /// Creates a decay configuration.
    pub const fn new(half_life_days: f32, base_weight: f32) -> Self {
        Self {
            half_life_days,
            base_weight,
        }
    }

    /// Decay constant `lambda = ln(2) / half_life`. A non-positive or
    /// non-finite half-life disables decay (`lambda = 0`).
    pub fn lambda(&self) -> f32 {
        if self.half_life_days.is_finite() && self.half_life_days > 0.0 {
            LN_2 / self.half_life_days
        } else {
            0.0
        }
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self::new(30.0, 1.0)
    }
}

#[inline]
fn clamp_exponent(x: f32) -> f32 {
    x.clamp(MIN_EXPONENT, 0.0)
}

/// Vector constants for [`exp_nonpositive`], built once per call: splatting
/// them inside the hot loop makes LLVM re-materialize each one through a
/// `memset_pattern16` call per iteration on aarch64.
struct ExpConsts {
    neg_lambda: f32x8,
    min_exponent: f32x8,
    log2_e: f32x8,
    ln2_hi: f32x8,
    ln2_lo: f32x8,
    poly: [f32x8; 5],
    bias: i32x8,
}

impl ExpConsts {
    fn new(lambda: f32) -> Self {
        Self {
            neg_lambda: f32x8::splat(-lambda),
            min_exponent: f32x8::splat(MIN_EXPONENT),
            log2_e: f32x8::splat(std::f32::consts::LOG2_E),
            ln2_hi: f32x8::splat(0.693_359_4),
            ln2_lo: f32x8::splat(-2.121_944_4e-4),
            poly: [
                f32x8::splat(1.0 / 720.0),
                f32x8::splat(1.0 / 120.0),
                f32x8::splat(1.0 / 24.0),
                f32x8::splat(1.0 / 6.0),
                f32x8::splat(0.5),
            ],
            bias: i32x8::splat(127),
        }
    }
}

/// `exp(x)` for `x` in `[MIN_EXPONENT, 0]`: Cody-Waite range reduction
/// `x = n*ln2 + r` with `|r| <= ln2/2`, a degree-6 polynomial for `exp(r)`
/// (truncation error below 2e-7 relative), and `2^n` assembled directly in
/// the exponent bits. Restricting the domain removes the overflow, NaN and
/// subnormal handling a general `exp` needs.
#[inline(always)]
fn exp_nonpositive(x: f32x8, k: &ExpConsts) -> f32x8 {
    let n = (x * k.log2_e).round();
    let r = n.mul_neg_add(k.ln2_hi, x);
    let r = n.mul_neg_add(k.ln2_lo, r);
    let p = k.poly[0]
        .mul_add(r, k.poly[1])
        .mul_add(r, k.poly[2])
        .mul_add(r, k.poly[3])
        .mul_add(r, k.poly[4])
        .mul_add(r, f32x8::ONE)
        .mul_add(r, f32x8::ONE);
    let biased: i32x8 = (n.round_int() + k.bias) << 23;
    p * bytemuck::cast::<i32x8, f32x8>(biased)
}

#[inline(always)]
fn decay_lanes(deltas: &[f32], k: &ExpConsts, out: &mut [f32]) {
    let lanes: [f32; LANES] = deltas.try_into().expect("exactly LANES deltas");
    let exponent = (f32x8::from(lanes) * k.neg_lambda)
        .max(k.min_exponent)
        .min(f32x8::ZERO);
    out.copy_from_slice(&exp_nonpositive(exponent, k).to_array());
}

/// Scalar reference implementation of `exp(-lambda * dt)`.
///
/// Negative deltas (timestamps in the future) are treated as zero age.
///
/// # Panics
/// Panics if `deltas_in_days` and `out_scores` differ in length.
pub fn compute_decay_scores_scalar(deltas_in_days: &[f32], lambda: f32, out_scores: &mut [f32]) {
    assert_eq!(
        deltas_in_days.len(),
        out_scores.len(),
        "deltas and output buffers must have equal length"
    );
    for (delta, out) in deltas_in_days.iter().zip(out_scores.iter_mut()) {
        *out = clamp_exponent(-lambda * delta).exp();
    }
}

/// SIMD implementation of `exp(-lambda * dt)` processing 8 lanes per step
/// (AVX on x86_64, paired NEON registers on aarch64), with a scalar tail.
///
/// # Panics
/// Panics if `deltas_in_days` and `out_scores` differ in length.
pub fn compute_decay_scores_simd(deltas_in_days: &[f32], lambda: f32, out_scores: &mut [f32]) {
    assert_eq!(
        deltas_in_days.len(),
        out_scores.len(),
        "deltas and output buffers must have equal length"
    );
    let k = ExpConsts::new(lambda);

    // Two independent vectors per iteration keep both FMA pipes busy.
    let mut in_pairs = deltas_in_days.chunks_exact(2 * LANES);
    let mut out_pairs = out_scores.chunks_exact_mut(2 * LANES);
    for (chunk, out_chunk) in (&mut in_pairs).zip(&mut out_pairs) {
        let (in_a, in_b) = chunk.split_at(LANES);
        let (out_a, out_b) = out_chunk.split_at_mut(LANES);
        decay_lanes(in_a, &k, out_a);
        decay_lanes(in_b, &k, out_b);
    }

    let mut in_chunks = in_pairs.remainder().chunks_exact(LANES);
    let mut out_chunks = out_pairs.into_remainder().chunks_exact_mut(LANES);
    for (chunk, out_chunk) in (&mut in_chunks).zip(&mut out_chunks) {
        decay_lanes(chunk, &k, out_chunk);
    }
    compute_decay_scores_scalar(in_chunks.remainder(), lambda, out_chunks.into_remainder());
}

/// Dispatches to the SIMD path, or to the scalar path when the
/// `scalar-fallback` feature is enabled.
#[inline]
pub fn compute_decay_scores(deltas_in_days: &[f32], lambda: f32, out_scores: &mut [f32]) {
    #[cfg(feature = "scalar-fallback")]
    compute_decay_scores_scalar(deltas_in_days, lambda, out_scores);
    #[cfg(not(feature = "scalar-fallback"))]
    compute_decay_scores_simd(deltas_in_days, lambda, out_scores);
}

#[inline]
fn age_in_days(current_time: i64, timestamp: i64) -> f32 {
    (current_time.saturating_sub(timestamp) as f64 * (1.0 / SECONDS_PER_DAY)) as f32
}

/// Computes `base_weight * exp(-lambda * age_days)` for every edge timestamp
/// (Unix seconds) relative to `current_time` (Unix seconds).
///
/// # Panics
/// Panics if `edge_timestamps` and `out_scores` differ in length.
pub fn batch_compute_decay(
    current_time: i64,
    edge_timestamps: &[i64],
    config: &DecayConfig,
    out_scores: &mut [f32],
) {
    assert_eq!(
        edge_timestamps.len(),
        out_scores.len(),
        "timestamps and output buffers must have equal length"
    );
    let lambda = config.lambda();
    let mut deltas = [0.0f32; CONVERSION_BLOCK];
    for (ts_block, out_block) in edge_timestamps
        .chunks(CONVERSION_BLOCK)
        .zip(out_scores.chunks_mut(CONVERSION_BLOCK))
    {
        let deltas = &mut deltas[..ts_block.len()];
        for (delta, &ts) in deltas.iter_mut().zip(ts_block) {
            *delta = age_in_days(current_time, ts);
        }
        compute_decay_scores(deltas, lambda, out_block);
        if config.base_weight != 1.0 {
            for score in out_block.iter_mut() {
                *score *= config.base_weight;
            }
        }
    }
}

/// Decay weight of a single timestamp; equivalent to a one-element
/// [`batch_compute_decay`] call.
pub fn decay_weight(current_time: i64, timestamp: i64, config: &DecayConfig) -> f32 {
    let exponent = clamp_exponent(-config.lambda() * age_in_days(current_time, timestamp));
    config.base_weight * exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    const DAY: i64 = 86_400;

    #[test]
    fn half_life_halves_weight() {
        let cfg = DecayConfig::new(10.0, 1.0);
        let w = decay_weight(10 * DAY, 0, &cfg);
        assert!((w - 0.5).abs() < 1e-6, "got {w}");
    }

    #[test]
    fn future_timestamps_are_zero_age() {
        let cfg = DecayConfig::new(10.0, 0.8);
        assert_eq!(decay_weight(0, 5 * DAY, &cfg), 0.8);
    }

    #[test]
    fn disabled_decay_returns_base_weight() {
        let cfg = DecayConfig::new(0.0, 0.7);
        let mut out = [0.0; 3];
        batch_compute_decay(1_000 * DAY, &[0, 10, 999 * DAY], &cfg, &mut out);
        assert_eq!(out, [0.7; 3]);
    }

    #[test]
    fn simd_matches_std_exp_on_10k_random_deltas() {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        let deltas: Vec<f32> = (0..10_000).map(|_| rng.gen_range(0.0..3_650.0)).collect();
        let lambda = DecayConfig::new(30.0, 1.0).lambda();

        let mut simd = vec![0.0; deltas.len()];
        compute_decay_scores_simd(&deltas, lambda, &mut simd);

        for (d, s) in deltas.iter().zip(&simd) {
            let reference = (-lambda * d).exp();
            assert!(
                (s - reference).abs() < 1e-5,
                "delta={d} simd={s} std={reference}"
            );
            if reference > 1e-30 {
                assert!(((s - reference) / reference).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn simd_handles_remainders_and_extremes() {
        let deltas = [0.0, 1.0, 1e9, -5.0, 3.5, 7.25, 100.0, 1e-3, 42.0, 0.5, 9.0];
        let lambda = 0.2;
        let mut simd = [0.0; 11];
        let mut scalar = [0.0; 11];
        compute_decay_scores_simd(&deltas, lambda, &mut simd);
        compute_decay_scores_scalar(&deltas, lambda, &mut scalar);
        for (a, b) in simd.iter().zip(&scalar) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(simd[3], 1.0);
        assert!(simd[2] < 1e-30);
    }

    #[test]
    fn batch_matches_single_weight() {
        let cfg = DecayConfig::new(7.0, 0.9);
        let now = 2_000 * DAY;
        let ts: Vec<i64> = (0..1_031).map(|i| now - i * 3_600 * 5).collect();
        let mut out = vec![0.0; ts.len()];
        batch_compute_decay(now, &ts, &cfg, &mut out);
        for (t, s) in ts.iter().zip(&out) {
            assert!((s - decay_weight(now, *t, &cfg)).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mismatched_lengths_panic() {
        let mut out = [0.0; 2];
        batch_compute_decay(0, &[1, 2, 3], &DecayConfig::default(), &mut out);
    }
}
