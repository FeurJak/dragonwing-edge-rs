//! Test data generators.
//!
//! Provides deterministic pseudo-random data for reproducible tests.

/// Simple linear congruential generator for deterministic pseudo-random f32s.
/// Not cryptographically secure, but fast and reproducible.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next u64.
    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate a random f32 in [min, max).
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Fill a slice with random f32s in [min, max).
    pub fn fill_f32(&mut self, dst: &mut [f32], min: f32, max: f32) {
        for v in dst.iter_mut() {
            *v = self.next_f32_range(min, max);
        }
    }
}

/// Generate test data for fill_f32.
pub fn fill_test_data(seed: u64) -> (usize, f32) {
    let mut rng = Rng::new(seed);
    let n = 1024 + (rng.next_u64() % 1024) as usize; // 1024..2048
    let value = rng.next_f32_range(-100.0, 100.0);
    (n, value)
}

/// Generate test data for axpy_f32.
pub fn axpy_test_data(seed: u64) -> (Vec<f32>, Vec<f32>, f32) {
    let mut rng = Rng::new(seed);
    let n = 1024 + (rng.next_u64() % 1024) as usize;
    let alpha = rng.next_f32_range(-10.0, 10.0);
    
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    rng.fill_f32(&mut x, -100.0, 100.0);
    rng.fill_f32(&mut y, -100.0, 100.0);
    
    (x, y, alpha)
}

/// Generate test data for relu_f32.
pub fn relu_test_data(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let n = 1024 + (rng.next_u64() % 1024) as usize;
    
    let mut x = vec![0.0f32; n];
    // Mix of positive and negative values
    rng.fill_f32(&mut x, -100.0, 100.0);
    x
}

/// Generate test data for gemm_f32.
pub fn gemm_test_data(seed: u64) -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
    let mut rng = Rng::new(seed);
    
    // Small-ish matrices to keep test time reasonable
    let m = 32 + (rng.next_u64() % 32) as usize; // 32..64
    let n = 32 + (rng.next_u64() % 32) as usize;
    let k = 32 + (rng.next_u64() % 32) as usize;
    
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    
    // Use smaller range to avoid large accumulation errors
    rng.fill_f32(&mut a, -1.0, 1.0);
    rng.fill_f32(&mut b, -1.0, 1.0);
    
    (a, b, m, n, k)
}
