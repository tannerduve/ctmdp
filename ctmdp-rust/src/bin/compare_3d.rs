use std::collections::HashMap;
use std::hash::Hash;

use ctmdp_rust::mdp::MDP;
use ctmdp_rust::pathmdp::{PathAction, PathState, PathWorld};
use ctmdp_rust::products::{BoxAction, BoxProduct, CartesianProduct, Product};
use ctmdp_rust::q_learning::q_learning;
use madepro::models::{ActionValue, Config};

type DeterministicPolicy<S, A> = HashMap<S, A>;

type BP3 = BoxProduct<BoxProduct<PathWorld, PathWorld>, PathWorld>;
type CP3 = CartesianProduct<CartesianProduct<PathWorld, PathWorld>, PathWorld>;
type BP3State = Product<Product<PathState, PathState>, PathState>;
type BP3Action = BoxAction<BoxAction<PathAction, PathAction>, PathAction>;
type CP3State = Product<Product<PathState, PathState>, PathState>;
type CP3Action = Product<Product<PathAction, PathAction>, PathAction>;

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

fn is_good_bp3_action(action: &BP3Action) -> bool {
    match action {
        BoxAction::Right(a3) => *a3 == PathAction::Next,
        BoxAction::Left(inner) => matches!(
            inner,
            BoxAction::Left(PathAction::Next) | BoxAction::Right(PathAction::Next)
        ),
    }
}

fn optimal_policy_bp3(bp: &BP3) -> DeterministicPolicy<BP3State, BP3Action> {
    let mut policy = HashMap::new();

    for state in bp.all_states().iter() {
        let actions = bp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen = actions[0].clone();
        for action in &actions {
            if is_good_bp3_action(action) {
                chosen = action.clone();
                break;
            }
        }

        policy.insert(state.clone(), chosen);
    }

    policy
}

fn optimal_policy_cp3(cp: &CP3) -> DeterministicPolicy<CP3State, CP3Action> {
    let mut policy = HashMap::new();

    let pair = Product::new(PathAction::Next, PathAction::Next);
    let target = Product::new(pair, PathAction::Next);

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

fn run_trial3<M>(
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
    println!("3D Gridworld-style Experiment: Box Product vs Cartesian Product");
    println!("======================================================================");

    let size = 4usize;
    let num_runs = 10usize;
    let num_episodes = 1500usize;

    let p1 = make_path_world(size);
    let p2 = make_path_world(size);
    let p3 = make_path_world(size);

    let bp_temp = BoxProduct::new(p1, p2);
    let bp: BP3 = BoxProduct::new(bp_temp, p3);

    let cp_temp = CartesianProduct::new(make_path_world(size), make_path_world(size));
    let cp: CP3 = CartesianProduct::new(cp_temp, make_path_world(size));

    analyze_action_space(&bp, "BP");
    analyze_action_space(&cp, "CP");

    let opt_bp = optimal_policy_bp3(&bp);
    let opt_cp = optimal_policy_cp3(&cp);

    println!();
    println!(
        "Running {} trials ({} episodes each)...",
        num_runs, num_episodes
    );

    let mut bp_dists = Vec::with_capacity(num_runs);
    let mut cp_dists = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        bp_dists.push(run_trial3(&bp, &opt_bp, num_episodes, size * 4));
        cp_dists.push(run_trial3(&cp, &opt_cp, num_episodes, size * 4));
    }

    let bp_mean: f64 = bp_dists.iter().copied().sum::<f64>() / bp_dists.len() as f64;
    let cp_mean: f64 = cp_dists.iter().copied().sum::<f64>() / cp_dists.len() as f64;

    println!();
    println!("Results:");
    println!("  BP policy distance: {:.4}", bp_mean);
    println!("  CP policy distance: {:.4}", cp_mean);
    println!("  Ratio (CP/BP): {:.2}x", cp_mean / bp_mean);

    println!();
    println!("======================================================================");
    println!("In 3D, CP still has a larger joint action space than BP.");
    println!("This experiment mirrors the Python 3D grid comparison.");
    println!("======================================================================");
}
