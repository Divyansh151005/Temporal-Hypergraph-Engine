//! Sinusoidal (Fourier) time encodings, identical to transformer positional
//! encodings but applied to epoch timestamps:
//!
//! * `TE(t, 2i)   = sin(t / 10000^(2i/d))`
//! * `TE(t, 2i+1) = cos(t / 10000^(2i/d))`

/// Base of the geometric frequency progression.
pub const FOURIER_BASE: f64 = 10_000.0;

/// Unit applied to epoch seconds before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeScale {
    /// Raw epoch seconds.
    Seconds,
    /// Epoch hours.
    #[default]
    Hours,
    /// Epoch days.
    Days,
}

impl TimeScale {
    /// Number of seconds in one unit of this scale.
    pub const fn seconds_per_unit(self) -> f64 {
        match self {
            TimeScale::Seconds => 1.0,
            TimeScale::Hours => 3_600.0,
            TimeScale::Days => 86_400.0,
        }
    }
}

/// Angles are evaluated in `f64`: epoch hours are ~5e5, which would lose most
/// of the phase precision of the high-frequency components in `f32`.
#[inline]
fn component(t: f64, j: usize, dim: usize) -> f32 {
    let i = (j / 2) as f64;
    let angle = t / FOURIER_BASE.powf(2.0 * i / dim as f64);
    if j % 2 == 0 {
        angle.sin() as f32
    } else {
        angle.cos() as f32
    }
}

/// Writes the `out.len()`-dimensional time encoding of `timestamp` (Unix
/// seconds) into `out`.
pub fn fourier_time_encoding(timestamp: i64, scale: TimeScale, out: &mut [f32]) {
    let dim = out.len();
    let t = timestamp as f64 / scale.seconds_per_unit();
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = component(t, j, dim);
    }
}

/// Allocating convenience wrapper around [`fourier_time_encoding`].
pub fn time_encoding(timestamp: i64, scale: TimeScale, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    fourier_time_encoding(timestamp, scale, &mut out);
    out
}

/// Adds `alpha * TE(timestamp)` to a semantic embedding in place.
pub fn add_time_encoding(semantic: &mut [f32], timestamp: i64, scale: TimeScale, alpha: f32) {
    let dim = semantic.len();
    let t = timestamp as f64 / scale.seconds_per_unit();
    for (j, slot) in semantic.iter_mut().enumerate() {
        *slot += alpha * component(t, j, dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_time_is_sin0_cos0() {
        let enc = time_encoding(0, TimeScale::Hours, 16);
        for pair in enc.chunks(2) {
            assert_eq!(pair, [0.0, 1.0]);
        }
    }

    #[test]
    fn matches_positional_encoding_closed_form() {
        // Position 5 of a 16-dim transformer positional encoding.
        let enc = time_encoding(5 * 3_600, TimeScale::Hours, 16);
        assert!((enc[0] - 5.0f32.sin()).abs() < 1e-6);
        assert!((enc[1] - 5.0f32.cos()).abs() < 1e-6);
        let f1 = 5.0f64 / 10_000f64.powf(2.0 / 16.0);
        assert!((enc[2] as f64 - f1.sin()).abs() < 1e-6);
        assert!((enc[3] as f64 - f1.cos()).abs() < 1e-6);
        let f7 = 5.0f64 / 10_000f64.powf(14.0 / 16.0);
        assert!((enc[14] as f64 - f7.sin()).abs() < 1e-6);
        assert!((enc[15] as f64 - f7.cos()).abs() < 1e-6);
    }

    #[test]
    fn components_lie_on_unit_circle() {
        let enc = time_encoding(1_760_000_000, TimeScale::Hours, 64);
        for pair in enc.chunks(2) {
            let norm = pair[0] * pair[0] + pair[1] * pair[1];
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn odd_dimension_ends_with_sine() {
        let enc = time_encoding(7 * 86_400, TimeScale::Days, 5);
        let f2 = 7.0f64 / 10_000f64.powf(4.0 / 5.0);
        assert!((enc[4] as f64 - f2.sin()).abs() < 1e-6);
    }

    #[test]
    fn add_time_encoding_is_scaled_sum() {
        let ts = 1_700_000_000;
        let mut semantic = vec![0.25f32; 32];
        add_time_encoding(&mut semantic, ts, TimeScale::Days, 0.5);
        let te = time_encoding(ts, TimeScale::Days, 32);
        for (s, t) in semantic.iter().zip(&te) {
            assert!((s - (0.25 + 0.5 * t)).abs() < 1e-6);
        }
    }
}
