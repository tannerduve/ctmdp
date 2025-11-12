use std::collections::HashMap;
use std::hash::Hash;

use crate::error::Error;

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Probability(f64);
impl Probability {
    pub const ZERO: Self = Probability(0.);
    pub const ONE: Self = Probability(1.);

    pub fn new(value: f64) -> Result<Self, Error> {
        if value < 0.0 || value > 1.0 {
            return Err(Error::ProbabilityOutOfRange);
        } else {
            Ok(Probability(value))
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
    pub fn complement(&self) -> Self {
        Probability(1.0 - self.0)
    }
    pub fn and(&self, other: Probability) -> Self {
        Probability(self.0 * other.0)
    }
    pub fn or(&self, other: Probability) -> Self {
        Probability((self.0 + other.0) - (self.0 * other.0))
    }
}

pub struct Measure<T> {
    dist: HashMap<T, Probability>,
}

impl<T: Eq + Hash> Measure<T> {
    pub fn from_distribution(dist: HashMap<T, Probability>) -> Result<Measure<T>, Error> {
        let sum: f64 = dist.values().map(|p| p.0).sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(Error::InvalidMeasure);
        } else {
            Ok(Measure { dist })
        }
    }

    pub fn deterministic(key: T) -> Measure<T> {
        let mut init = HashMap::new();
        init.insert(key, Probability::ONE);
        Self::from_distribution(init).unwrap()
    }

    pub fn dist(&self) -> &HashMap<T, Probability> {
        &self.dist
    }
    pub fn get_prob(&self, key: &T) -> Option<&Probability> {
        self.dist.get(key)
    }
}
