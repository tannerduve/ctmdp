use std::collections::HashMap;
use std::hash::Hash;

use ctmdp_rust::mdp::MDP;
use ctmdp_rust::pathmdp::{PathAction, PathState, PathWorld};
use ctmdp_rust::products::{BoxAction, BoxProduct, CartesianProduct, Product};
use ctmdp_rust::q_learning::q_learning;
use madepro::models::{ActionValue, Config};

type DeterministicPolicy<S, A> = HashMap<S, A>;

fn make_path_world(length: usize) -> PathWorld {
    let states: Vec<PathState> = (0..length).map(PathState::new).collect();
    let actions = vec![PathAction::Next, PathAction::Prev];
    PathWorld::new(states, actions)
}

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
    bp: &BoxProduct<PathWorld, PathWorld>,
) -> DeterministicPolicy<Product<PathState, PathState>, BoxAction<PathAction, PathAction>>
{
    let mut policy = HashMap::new();

    for state in bp.all_states().iter() {
        let actions = bp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen = actions[0].clone();
        for action in actions {
            match action {
                BoxAction::Left(PathAction::Next) | BoxAction::Right(PathAction::Next) => {
                    chosen = action.clone();
                    break;
                }
                _ => {}
            }
        }

        policy.insert(state.clone(), chosen);
    }

    policy
}

fn optimal_policy_cp(
    cp: &CartesianProduct<PathWorld, PathWorld>,
) -> DeterministicPolicy<Product<PathState, PathState>, Product<PathAction, PathAction>>
{
    let mut policy = HashMap::new();
    let target = Product::new(PathAction::Next, PathAction::Next);

    for state in cp.all_states().iter() {
        let actions = cp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen = actions[0].clone();
        for action in actions {
            if action == target {
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

fn analyze_action_space_bp_cp(
    bp: &BoxProduct<PathWorld, PathWorld>,
    cp: &CartesianProduct<PathWorld, PathWorld>,
) {
    let bp_states: Vec<_> = bp.all_states().iter().collect();
    let cp_states: Vec<_> = cp.all_states().iter().collect();

    let bp_actions: usize = bp_states.iter().map(|s| bp.actions_at(s).len()).sum();
    let cp_actions: usize = cp_states.iter().map(|s| cp.actions_at(s).len()).sum();

    let bp_avg = bp_actions as f64 / bp_states.len() as f64;
    let cp_avg = cp_actions as f64 / cp_states.len() as f64;

    println!("  States: BP={}, CP={}", bp_states.len(), cp_states.len());
    println!("  Avg actions/state: BP={:.1}, CP={:.1}", bp_avg, cp_avg);
}

fn run_experiment(size: usize, num_runs: usize, num_episodes: usize) {
    println!();
    println!("======================================================================");
    println!("Box vs Cartesian Product on {}x{} chain", size, size);
    println!("======================================================================");

    let bp = BoxProduct::new(make_path_world(size), make_path_world(size));
    let cp = CartesianProduct::new(make_path_world(size), make_path_world(size));

    analyze_action_space_bp_cp(&bp, &cp);

    let opt_bp = optimal_policy_bp(&bp);
    let opt_cp = optimal_policy_cp(&cp);

    let mut bp_dists = Vec::with_capacity(num_runs);
    let mut cp_dists = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        bp_dists.push(run_trial(&bp, &opt_bp, num_episodes, size * 3));
        cp_dists.push(run_trial(&cp, &opt_cp, num_episodes, size * 3));
    }

    let bp_mean: f64 = bp_dists.iter().copied().sum::<f64>() / bp_dists.len() as f64;
    let cp_mean: f64 = cp_dists.iter().copied().sum::<f64>() / cp_dists.len() as f64;

    println!("Results after {} episodes:", num_episodes);
    println!("  BP policy distance: {:.4}", bp_mean);
    println!("  CP policy distance: {:.4}", cp_mean);
    println!("  Ratio (CP/BP): {:.2}x", cp_mean / bp_mean);
}

fn main() {
    println!("2D chain: Box vs Cartesian products");

    let configs = vec![(4usize, 10usize, 1500usize), (6, 10, 2000)];

    for (size, num_runs, num_episodes) in configs {
        run_experiment(size, num_runs, num_episodes);
    }
}
