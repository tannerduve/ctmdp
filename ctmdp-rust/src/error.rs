use core::error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Probability must be between in 0 and 1 inclusive")]
    ProbabilityOutOfRange,
}
