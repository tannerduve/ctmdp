use super::{END_TRANSITION_REWARD, NO_OP_TRANSITION_REWARD};
use crate::measure::Measure;
use crate::{mdp::MDP, measure::Probability};
use madepro::models::{Action, Sampler, State};
use std::{collections::HashMap, hash::Hash, path::Path, vec};

#[derive(Debug)]
pub struct BoxProduct<M1: MDP, M2: MDP>
where
    M1::State: Clone,
    M2::State: Clone,
{
    mdp1: M1,
    mdp2: M2,
    states: Sampler<Product<M1::State, M2::State>>,
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

impl<S1, S2> Product<S1, S2> {
    pub fn new(s1: S1, s2: S2) -> Self {
        Product {fst: s1, snd: s2}
    }
}

impl<S1: State, S2: State> State for Product<S1, S2> {}

impl<A1: Action, A2: Action> Action for BoxAction<A1, A2> {}

impl<M1: MDP, M2: MDP> BoxProduct<M1, M2>
where
    M1::State: Clone,
    M2::State: Clone,
{
    pub fn new(mdp1: M1, mdp2: M2) -> Self {
        let s1 = mdp1.all_states().iter();
        let s2 = mdp2.all_states().iter();
        let states: Vec<Product<M1::State, M2::State>> = s1
            .zip(s2)
            .map(|(x1, x2)| Product::new(x1.clone(), x2.clone()))
            .collect();
        let states = Sampler::new(states);
        
        BoxProduct {
            mdp1,
            mdp2,
            states,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct CartesianProduct<M1, M2> {
    mdp1: M1,
    mdp2: M2,
}

//TODO:
impl<M1, M2> MDP for BoxProduct<M1, M2>
where
    M1: MDP,
    M2: MDP,
    M1::State: Clone,
    M2::State: Clone,
    M1::Action: Clone,
    M2::Action: Clone,
{
    type State = Product<M1::State, M2::State>;
    type Action = BoxAction<M1::Action, M2::Action>;

    fn all_states(&self) -> &Sampler<Self::State> {
        &self.states
    }

    fn actions_at(&self, state: &Self::State) -> Vec<Self::Action> {
        // TODO: Get actions available at state.fst from mdp1
        // Hint: use self.mdp1.actions_at(...)
        let left_actions = todo!();
        
        // TODO: Get actions available at state.snd from mdp2
        let right_actions = todo!();
        
        // TODO: Combine left_actions and right_actions into a single Vec
        // Hint: you need to wrap them in BoxAction::Left and BoxAction::Right
        // and then merge the two collections
        todo!()
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        // TODO: When is a product state (s1, s2) final?
        // Think about the relationship between the two components
        todo!()
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> (Measure<Self::State>, f64) {
        match action {
            BoxAction::Left(a1) => {
                // TODO: Apply action a1 to the first component
                // What happens to the second component?
                // How do you transform the resulting measure into product states?
                todo!()
            }
            BoxAction::Right(a2) => {
                // TODO: Apply action a2 to the second component
                // What happens to the first component?
                // How do you transform the resulting measure into product states?
                todo!()
            }
        }
    }
}
