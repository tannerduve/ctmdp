pub mod constructors;
pub mod error;
pub mod gridworld;
pub mod mdp;
pub mod mdp_trait;
pub mod measure;
pub mod products;
pub mod q_learning;
pub mod pathmdp;

const NO_OP_TRANSITION_REWARD: f64 = -1.0;
const END_TRANSITION_REWARD: f64 = 10.0;
