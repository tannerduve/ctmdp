use std::collections::HashMap;
use std::hash::Hash;

use ctmdp_rust::error::Error;
use ctmdp_rust::mdp::MDP;
use ctmdp_rust::measure::Measure;
use ctmdp_rust::products::{BoxAction, BoxProduct, CartesianProduct, Product};
use ctmdp_rust::q_learning::q_learning;
use madepro::models::{Action, ActionValue, Config, Sampler, State};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ChainState(pub usize);

impl State for ChainState {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ChainAction {
    Next,
    Prev,
    Detour,
}

impl Action for ChainAction {}

struct BranchedChain {
    states: Sampler<ChainState>,
    length: usize,
    branch_states: Vec<usize>,
}

impl BranchedChain {
    fn new(length: usize, branch_states: Vec<usize>) -> Self {
        let states: Vec<ChainState> = (0..length).map(ChainState).collect();
        BranchedChain {
            states: states.into(),
            length,
            branch_states,
        }
    }

    fn is_branch_state(&self, index: usize) -> bool {
        self.branch_states.contains(&index)
    }
}

impl MDP for BranchedChain {
    type State = ChainState;
    type Action = ChainAction;

    fn all_states(&self) -> &Sampler<Self::State> {
        &self.states
    }

    fn actions_at(&self, state: &Self::State) -> Vec<Self::Action> {
        let mut actions = vec![ChainAction::Prev, ChainAction::Next];
        if self.is_branch_state(state.0) {
            actions.push(ChainAction::Detour);
        }
        actions
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        state.0 == self.length - 1
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(Measure<Self::State>, f64), Error> {
        let current = state.0;
        let next = match action {
            ChainAction::Next => usize::min(current + 1, self.length - 1),
            ChainAction::Prev => current.saturating_sub(1),
            ChainAction::Detour => 0,
        };

        let mut reward = match action {
            ChainAction::Next => 0.1,
            ChainAction::Prev => -0.5,
            ChainAction::Detour => -2.0,
        };

        if next == self.length - 1 {
            reward += 10.0;
        }

        let measure = Measure::deterministic(ChainState(next));
        Ok((measure, reward))
    }
}

type DeterministicPolicy<S, A> = HashMap<S, A>;

fn greedy_policy<M>(
    mdp: &M,
    q_values: &ActionValue<M::State, M::Action>,
) -> DeterministicPolicy<M::State, M::Action>
where
    M: MDP,
    M::State: Clone + Eq + Hash,
    M::Action: Clone + Eq + Hash,
{
    let mut policy = HashMap::new();

    for state in mdp.all_states().iter() {
        let actions = mdp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut best_action = actions[0].clone();
        let mut best_value = q_values.get(state, &best_action);

        for action in actions.into_iter().skip(1) {
            let value = q_values.get(state, &action);
            if value > best_value {
                best_value = value;
                best_action = action.clone();
            }
        }

        policy.insert(state.clone(), best_action);
    }

    policy
}

fn policy_distance<S, A>(
    learned: &DeterministicPolicy<S, A>,
    optimal: &DeterministicPolicy<S, A>,
) -> f64
where
    S: Eq + Hash,
    A: Eq,
{
    let mut total = 0.0;
    let mut count = 0.0;

    for (state, opt_action) in optimal.iter() {
        if let Some(learned_action) = learned.get(state) {
            count += 1.0;
            if learned_action != opt_action {
                total += 1.0;
            }
        }
    }

    if count == 0.0 {
        1.0
    } else {
        total / count
    }
}

fn optimal_policy_bp(
    bp: &BoxProduct<BranchedChain, BranchedChain>,
) -> DeterministicPolicy<Product<ChainState, ChainState>, BoxAction<ChainAction, ChainAction>>
{
    let mut policy = HashMap::new();

    for state in bp.all_states().iter() {
        let actions = bp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen: Option<BoxAction<ChainAction, ChainAction>> = None;

        // Prefer "Next" actions
        for action in &actions {
            match action {
                BoxAction::Left(ChainAction::Next) | BoxAction::Right(ChainAction::Next) => {
                    chosen = Some(action.clone());
                    break;
                }
                _ => {}
            }
        }

        // If no "Next", fall back to "Prev" and avoid detours
        if chosen.is_none() {
            for action in &actions {
                match action {
                    BoxAction::Left(ChainAction::Prev) | BoxAction::Right(ChainAction::Prev) => {
                        chosen = Some(action.clone());
                        break;
                    }
                    _ => {}
                }
            }
        }

        let chosen = chosen.unwrap_or_else(|| actions[0].clone());
        policy.insert(state.clone(), chosen);
    }

    policy
}

fn optimal_policy_cp(
    cp: &CartesianProduct<BranchedChain, BranchedChain>,
) -> DeterministicPolicy<Product<ChainState, ChainState>, Product<ChainAction, ChainAction>>
{
    let mut policy = HashMap::new();
    let target = Product::new(ChainAction::Next, ChainAction::Next);

    for state in cp.all_states().iter() {
        let actions = cp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen = actions[0].clone();
        for action in &actions {
            if *action == target {
                chosen = action.clone();
                break;
            }
        }

        policy.insert(state.clone(), chosen);
    }

    policy
}

fn run_trial<M>(
    mdp: &M,
    optimal_policy: &DeterministicPolicy<M::State, M::Action>,
    num_episodes: usize,
    max_steps: usize,
) -> f64
where
    M: MDP,
    M::State: Clone + Eq + Hash,
    M::Action: Clone + Eq + Hash,
{
    let mut config = Config::default();
    config.num_episodes = num_episodes as u32;
    config.max_num_steps = max_steps as u32;
    config.learning_rate = 0.1;
    config.discount_factor = 0.9;
    config.exploration_rate = 0.1;

    let q_values = q_learning(mdp, &config).expect("q_learning failed");
    let learned = greedy_policy(mdp, &q_values);
    policy_distance(&learned, optimal_policy)
}

fn analyze_action_space<M>(mdp: &M, label: &str)
where
    M: MDP,
{
    let states: Vec<_> = mdp.all_states().iter().collect();
    let total_actions: usize = states.iter().map(|s| mdp.actions_at(s).len()).sum();
    let avg = total_actions as f64 / states.len() as f64;

    println!("  {} states: {}", label, states.len());
    println!("  {} avg actions/state: {:.1}", label, avg);
}

fn main() {
    println!("======================================================================");
    println!("Chain with Branches Experiment: Box Product vs Cartesian Product");
    println!("======================================================================");

    let configs = vec![(6usize, vec![2usize, 4usize]), (8usize, vec![3usize, 5usize])];
    let num_runs = 10usize;

    for (length, branches) in configs {
        println!();
        println!("Chain length {}, branches at {:?}", length, branches);

        let chain1 = BranchedChain::new(length, branches.clone());
        let chain2 = BranchedChain::new(length, branches.clone());

        let bp = BoxProduct::new(chain1, chain2);
        let cp = CartesianProduct::new(
            BranchedChain::new(length, branches.clone()),
            BranchedChain::new(length, branches.clone()),
        );

        analyze_action_space(&bp, "BP");
        analyze_action_space(&cp, "CP");

        let opt_bp = optimal_policy_bp(&bp);
        let opt_cp = optimal_policy_cp(&cp);

        let mut bp_dists = Vec::with_capacity(num_runs);
        let mut cp_dists = Vec::with_capacity(num_runs);

        for _ in 0..num_runs {
            bp_dists.push(run_trial(&bp, &opt_bp, 1000, length * 3));
            cp_dists.push(run_trial(&cp, &opt_cp, 1000, length * 3));
        }

        let bp_mean: f64 = bp_dists.iter().copied().sum::<f64>() / bp_dists.len() as f64;
        let cp_mean: f64 = cp_dists.iter().copied().sum::<f64>() / cp_dists.len() as f64;

        println!("  BP distance: {:.4}", bp_mean);
        println!("  CP distance: {:.4}", cp_mean);
        println!("  Ratio (CP/BP): {:.2}x", cp_mean / bp_mean);
    }

    println!();
    println!("======================================================================");
    println!("Heterogeneous action spaces: detours exist only at some states.");
    println!("BP can learn to avoid detours independently per component.");
    println!("======================================================================");
}
