use super::{END_TRANSITION_REWARD, NO_OP_TRANSITION_REWARD};
use crate::measure::Measure;
use crate::{mdp::MDP, measure::Probability};
use crate::error::Error;
use madepro::models::{Action, Sampler, State};
use std::{collections::HashMap, hash::Hash, path::Path, vec};

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum PathAction {
    Next,
    Prev,
}

impl Action for PathAction {}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct PathState(pub(crate) usize);

impl PathState {
    pub fn new(idx: usize) -> Self {
        PathState(idx)
    }
}

impl State for PathState {}

pub struct PathWorld {
    states: Sampler<PathState>,
    actions: Sampler<PathAction>,
    // cyclic: bool
}

impl PathWorld {
    pub fn new(states: Vec<PathState>, actions: Vec<PathAction>) -> Self {
        PathWorld {
            states: states.into(),
            actions: actions.into(),
        }
    }
    // pub fn cyclic_new(length: usize) -> Self {
    //     PathWorld { length, actions: vec![PathAction::Next, PathAction::Prev].into(), cyclic: true }
    // }
    pub fn length(&self) -> usize {
        self.states.iter().count() // Compute from states
    }
}

impl MDP for PathWorld {
    type State = PathState;
    type Action = PathAction;

    fn all_states(&self) -> &Sampler<Self::State> {
        &self.states
    }

    fn is_final_state(&self, state: &PathState) -> bool {
        state.0 == self.length() - 1
    }

    fn actions_at(&self, _state: &Self::State) -> Vec<Self::Action> {
        self.actions.iter().cloned().collect()
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(Measure<Self::State>, f64), Error> {
        let current = state.0;
        let length = self.length();
        // Tentative position
        let next = match action {
            Self::Action::Next => current + 1,
            Self::Action::Prev => current.saturating_sub(1),
        };
        // Out of bounds check
        if next >= length || next == current {
            let measure = Measure::deterministic(state.clone());
            return Ok((measure, NO_OP_TRANSITION_REWARD));
        }
        let reward = match action {
            Self::Action::Next => 0.1,
            Self::Action::Prev => -0.5,
        };
        let measure = Measure::deterministic(PathState(next));
        if next == length - 1 {
            Ok((measure, END_TRANSITION_REWARD + reward))
        } else {
            Ok((measure, reward))
        }
    }
}
