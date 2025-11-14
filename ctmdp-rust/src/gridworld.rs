use crate::{mdp, measure::Measure};
use madepro::environments::gridworld::Gridworld;
use madepro::environments::gridworld::{GridworldAction, GridworldState};
use madepro::models::{MDP, Sampler};
use std::ops::Deref;

impl mdp::MDP for Gridworld {
    type State = GridworldState;
    type Action = GridworldAction;

    fn all_states(&self) -> &Sampler<Self::State> {
        self.get_states()
    }

    fn all_actions(&self) -> &Sampler<Self::Action> {
        self.get_actions()
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        self.is_state_terminal(state)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> (Measure<Self::State>, f64) {
        let (transition_state, reward) = self.transition(state, action);
        let map = Measure::deterministic(transition_state);
        (map, reward)
    }
}

pub struct GridworldWithGoals {
    gridworld: Gridworld,
    goal_states: Vec<GridworldState>,
}

impl Deref for GridworldWithGoals {
    type Target = Gridworld;

    fn deref(&self) -> &Self::Target {
        &self.gridworld
    }
}

impl From<Gridworld> for GridworldWithGoals {
    fn from(gridworld: Gridworld) -> Self {
        GridworldWithGoals {
            gridworld,
            goal_states: Vec::new(), // Empty by default
        }
    }
}

impl GridworldWithGoals {
    pub fn new(gridworld: Gridworld, goal_states: Vec<GridworldState>) -> Self {
        GridworldWithGoals {
            gridworld,
            goal_states,
        }
    }
    pub fn get_goals(&self) -> &Vec<GridworldState> {
        &self.goal_states
    }
}

impl mdp::MDP for GridworldWithGoals {
    type State = GridworldState;
    type Action = GridworldAction;

    fn all_states(&self) -> &Sampler<Self::State> {
        self.get_states()
    }

    fn all_actions(&self) -> &Sampler<Self::Action> {
        self.get_actions()
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        self.is_state_terminal(state)
    }

    fn is_goal(&self, st: &Self::State) -> bool {
        self.goal_states.contains(st)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> (Measure<Self::State>, f64) {
        self.gridworld.stochastic_transition(state, action)
    }
}
