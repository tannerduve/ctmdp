use crate::measure::Measure;
use crate::error::Error;
use madepro::models::{Action, Sampler, State};

pub trait MDP {
    type State: State;
    type Action: Action;

    fn all_states(&self) -> &Sampler<Self::State>;

    fn actions_at(&self, state: &Self::State) -> Vec<Self::Action>;

    fn is_final_state(&self, st: &Self::State) -> bool;

    fn is_goal(&self, st: &Self::State) -> bool {
        Self::is_final_state(self, st)
    }

    fn all_state_action_pairs(&self) -> Vec<(Self::State, Self::Action)> {
        self.all_states()
            .iter()
            .flat_map(|s| self.actions_at(s).into_iter().map(move |a| (s.clone(), a)))
            .collect()
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(Measure<Self::State>, f64), Error>;
}
