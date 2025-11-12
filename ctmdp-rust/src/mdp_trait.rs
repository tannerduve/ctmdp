use crate::measure::Measure;
use madepro::models::{Action, Sampler, State};

pub trait MDP {
    type State: State;
    type Action: Action;

    fn all_states(&self) -> &Sampler<Self::State>;

    fn all_actions(&self) -> &Sampler<Self::Action>;

    fn is_final_state(&self, st: &Self::State) -> bool;

    fn is_goal(&self, st: &Self::State) -> bool {
        Self::is_final_state(self, st)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> (Measure<Self::State>, f64);
}
