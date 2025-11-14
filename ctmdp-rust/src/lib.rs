pub mod error;
pub mod gridworld;
pub mod mdp;
pub mod measure;
pub mod pathmdp;
pub mod products;
pub mod q_learning;

const NO_OP_TRANSITION_REWARD: f64 = -1.0;
const END_TRANSITION_REWARD: f64 = 10.0;
