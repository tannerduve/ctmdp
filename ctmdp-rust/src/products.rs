use crate::measure::Measure;
use crate::{mdp::MDP, measure::Probability};
use crate::error::Error;
use madepro::models::{Action, Sampler, State};
use std::{collections::HashMap, hash::Hash};

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
        Product { fst: s1, snd: s2 }
    }
}

impl<S1: State, S2: State> State for Product<S1, S2> {}

impl<A1: Action, A2: Action> Action for Product<A1, A2> {}

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

        BoxProduct { mdp1, mdp2, states }
    }
}

#[derive(Debug)]
pub struct CartesianProduct<M1:MDP, M2:MDP> {
    mdp1: M1,
    mdp2: M2,
    states: Sampler<Product<M1::State, M2::State>>,
}

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
        let left_actions: Vec<BoxAction<M1::Action, M2::Action>> = self
            .mdp1
            .actions_at(&state.fst)
            .iter()
            .map(|a: &<M1 as MDP>::Action| BoxAction::<M1::Action, M2::Action>::Left(a.clone()))
            .collect();

        let right_actions: Vec<BoxAction<M1::Action, M2::Action>> = self
            .mdp2
            .actions_at(&state.snd)
            .iter()
            .map(|a: &<M2 as MDP>::Action| BoxAction::<M1::Action, M2::Action>::Right(a.clone()))
            .collect();

        [left_actions, right_actions].concat()
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        self.mdp1.is_final_state(&state.fst) && self.mdp2.is_final_state(&state.snd)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(Measure<Self::State>, f64), Error> {
        match action {
            BoxAction::Left(a1) => {
                let (measure1, prob1) = self.mdp1.stochastic_transition(&state.fst, a1)?;
                let measure2 = Measure::deterministic(state.snd.clone());
                let product_dist: HashMap<Product<M1::State, M2::State>, Probability> = measure1.product(&measure2)
                    .unwrap()
                    .dist()
                    .iter()
                    .map(|((s1, s2), prob)| (Product::new(s1.clone(), s2.clone()), *prob))
                    .collect();
                let product_measure = Measure::from_distribution(product_dist)?;
                Ok((product_measure, prob1))
            }
            BoxAction::Right(a2) => {
                let (measure2, prob2) = self.mdp2.stochastic_transition(&state.snd, a2)?;
                let measure1 = Measure::deterministic(state.fst.clone());
                let product_dist: HashMap<Product<M1::State, M2::State>, Probability> = measure1.product(&measure2)
                    .unwrap()
                    .dist()
                    .iter()
                    .map(|((s1, s2), prob)| (Product::new(s1.clone(), s2.clone()), *prob))
                    .collect();
                let product_measure = Measure::from_distribution(product_dist)?;
                Ok((product_measure, prob2))
            }
        }
    }
}
impl<M1, M2> CartesianProduct<M1, M2>
where
    M1: MDP,
    M2: MDP,
    M1::State: Clone,
    M2::State: Clone,
{
    pub fn new(mdp1: M1, mdp2: M2) -> Self {
        let mut states = Vec::new();

        for s1 in mdp1.all_states().iter() {
            for s2 in mdp2.all_states().iter() {
                states.push(Product::new(s1.clone(), s2.clone()));
            }
        }
        let states = Sampler::new(states);

        CartesianProduct { mdp1, mdp2, states }
    }
}

impl<M1, M2> MDP for CartesianProduct<M1, M2>
where
    M1: MDP,
    M2: MDP,
    M1::State: Clone,
    M2::State: Clone,
    M1::Action: Clone,
    M2::Action: Clone,
{
    type State = Product<M1::State, M2::State>;
    type Action = Product<M1::Action, M2::Action>;
    fn all_states(&self) -> &Sampler<Self::State> {
        &self.states
    }

    fn actions_at(&self, state: &Self::State) -> Vec<Self::Action> {
        let actions1 = self.mdp1.actions_at(&state.fst);
        let actions2 = self.mdp2.actions_at(&state.snd);

        let mut out = Vec::with_capacity(actions1.len() * actions2.len());
        for a1 in actions1 {
            for a2 in actions2.iter() {
                out.push(Product::new(a1.clone(), a2.clone()));
            }
        }
        out
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        self.mdp1.is_final_state(&state.fst) && self.mdp2.is_final_state(&state.snd)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(Measure<Self::State>, f64), Error> {
        // product field is `.fst` and `.snd`
        let (m1, r1) = self.mdp1.stochastic_transition(&state.fst, &action.fst)?;
        let (m2, r2) = self.mdp2.stochastic_transition(&state.snd, &action.snd)?;

        let joint = m1.product(&m2)?;
        let dist = joint
            .dist()
            .iter()
            .map(|((s1, s2), p)| (Product::new(s1.clone(), s2.clone()), *p))
            .collect();

        Ok((Measure::from_distribution(dist)?, r1 + r2))
    }

}

