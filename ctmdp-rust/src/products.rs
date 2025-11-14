use super::{END_TRANSITION_REWARD, NO_OP_TRANSITION_REWARD};
use crate::measure::Measure;
use crate::{mdp_trait::MDP, measure::Probability};
use madepro::models::{Action, Sampler, State};
use std::{collections::HashMap, hash::Hash, path::Path, vec};

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct BoxProduct<M1, M2> {
    mdp1: M1,
    mdp2: M2,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum BoxAction<A1, A2> {
    Left(A1),
    Right(A2),
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct CartesianAction<A1, A2> {
    action1: A1,
    action2: A2,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct Product<S1, S2> {
    fst: S1,
    snd: S2,
}

impl<S1: State, S2: State> State for Product<S1, S2> {}

impl<A1: Action, A2: Action> Action for BoxAction<A1, A2> {}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct CartesianProduct<M1, M2> {
    mdp1: M1,
    mdp2: M2,
}

//TODO:
// impl<M1: MDP, M2: MDP> MDP for BoxProduct<M1, M2> {
//     type State = Product<M1::State, M2::State>;
//     type Action = BoxAction<M1::Action, M2::Action>;

//     fn stochastic_transition(...) -> ... {
//         // Implement product logic here
//     }
// }
