use crate::measure::Probability;
// use rand::seq::SliceRandom;
// use rand::thread_rng;
use std::collections::HashMap;
use std::hash::Hash;
// use std::os::macos::raw::stat;

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateLabel(pub(crate) String);

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionLabel(pub(crate) String);

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MDPLabel(pub(crate) String);

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug)]
enum ActionData {
    Deterministic(StateLabel),
    DeterministicWithReward(StateLabel, f64),
    Stochastic(HashMap<StateLabel, Probability>),
    StochasticWithReward(HashMap<StateLabel, Probability>, f64),
}

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
type Description = HashMap<StateLabel, HashMap<ActionLabel, ActionData>>;

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug)]
pub struct Action {
    state_label: StateLabel,
    label: ActionLabel,
    reward: f64,
    measure: Probability,
}

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug)]
pub struct State {
    label: StateLabel,
    actions: HashMap<ActionLabel, Action>,
}

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
impl State {
    pub fn new(label: StateLabel) -> Self {
        State {
            label,
            actions: HashMap::new(),
        }
    }
}

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
#[derive(Debug)]
pub struct MDP {
    states: HashMap<StateLabel, State>,
    start: Option<StateLabel>,
    goals: Vec<StateLabel>,
}

// TODO: Remove #[allow(dead_code)] when MDP implementation is complete
#[allow(dead_code)]
impl MDP {
    // TODO - fill in
    // pub fn from_description(label: MDPLabel, description: Description) -> Self {
    //     let states = description.iter()
    //     .map(|(state_label, action_map)| {
    //         action_map.iter()
    //         .map(|(action_label, action_data)| {
    //             match action_data {
    //                 ActionData::Deterministic(st_label) => {

    //                 }
    //             }

    //         })
    //         .collect()
    //     // Build complete State here, including actions
    //     // Hint: actions can also be built with .iter().map().collect()
    //     })
    //     .collect();

    //     MDP { states, start: Option::None, goals: Vec::new()  }
    // }
}
