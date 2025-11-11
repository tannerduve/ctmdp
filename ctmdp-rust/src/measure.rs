use std::collections::HashMap;
use std::hash::Hash;

use crate::error::Error;

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Probability(f64);
impl Probability {
    pub const ZERO: Self = Probability(0.);

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
    pub fn new() -> Self {
        Measure {
            dist: HashMap::new(),
        }
    }
    pub fn dist(&self) -> &HashMap<T, Probability> {
        &self.dist
    }
    pub fn get_prob(&self, key: &T) -> Option<&Probability> {
        self.dist.get(key)
    }
    pub fn add_probs(&self) -> Result<Probability, Error> {
        Probability::new(self.dist.values().map(|p| p.0).sum())
    }
}
