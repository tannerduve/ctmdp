use std::collections::HashMap;
use std::hash::Hash;

use ctmdp_rust::mdp::MDP;
use ctmdp_rust::pathmdp::{PathAction, PathState, PathWorld};
use ctmdp_rust::products::{BoxAction, BoxProduct, CartesianProduct, Product};
use ctmdp_rust::q_learning::q_learning;
use madepro::models::{ActionValue, Config};

type DeterministicPolicy<S, A> = HashMap<S, A>;

// 6D box and cartesian products built by iteratively composing 1D paths.
type BP2 = BoxProduct<PathWorld, PathWorld>;
type BP3 = BoxProduct<BP2, PathWorld>;
type BP4 = BoxProduct<BP3, PathWorld>;
type BP5 = BoxProduct<BP4, PathWorld>;
type BP6 = BoxProduct<BP5, PathWorld>;

type CP2 = CartesianProduct<PathWorld, PathWorld>;
type CP3 = CartesianProduct<CP2, PathWorld>;
type CP4 = CartesianProduct<CP3, PathWorld>;
type CP5 = CartesianProduct<CP4, PathWorld>;
type CP6 = CartesianProduct<CP5, PathWorld>;

type P2 = Product<PathState, PathState>;
type P3 = Product<P2, PathState>;
type P4 = Product<P3, PathState>;
type P5 = Product<P4, PathState>;
type BP6State = Product<P5, PathState>;
type BP2Action = BoxAction<PathAction, PathAction>;
type BP3Action = BoxAction<BP2Action, PathAction>;
type BP4Action = BoxAction<BP3Action, PathAction>;
type BP5Action = BoxAction<BP4Action, PathAction>;
type BP6Action = BoxAction<BP5Action, PathAction>;

type CP6State = BP6State;

type A2 = Product<PathAction, PathAction>;
type A3 = Product<A2, PathAction>;
type A4 = Product<A3, PathAction>;
type A5 = Product<A4, PathAction>;
type CP6Action = Product<A5, PathAction>;

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

fn has_next_bp2(action: &BP2Action) -> bool {
    match action {
        BoxAction::Left(a1) | BoxAction::Right(a1) => *a1 == PathAction::Next,
    }
}

fn has_next_bp3(action: &BP3Action) -> bool {
    match action {
        BoxAction::Right(a3) => *a3 == PathAction::Next,
        BoxAction::Left(inner2) => has_next_bp2(inner2),
    }
}

fn has_next_bp4(action: &BP4Action) -> bool {
    match action {
        BoxAction::Right(a4) => *a4 == PathAction::Next,
        BoxAction::Left(inner3) => has_next_bp3(inner3),
    }
}

fn has_next_bp5(action: &BP5Action) -> bool {
    match action {
        BoxAction::Right(a5) => *a5 == PathAction::Next,
        BoxAction::Left(inner4) => has_next_bp4(inner4),
    }
}

fn has_next_bp6(action: &BP6Action) -> bool {
    match action {
        BoxAction::Right(a6) => *a6 == PathAction::Next,
        BoxAction::Left(inner5) => has_next_bp5(inner5),
    }
}

fn optimal_policy_bp6(bp: &BP6) -> DeterministicPolicy<BP6State, BP6Action> {
    let mut policy = HashMap::new();

    for state in bp.all_states().iter() {
        let actions = bp.actions_at(state);
        if actions.is_empty() {
            continue;
        }

        let mut chosen = actions[0].clone();
        for action in &actions {
            if has_next_bp6(action) {
                chosen = action.clone();
                break;
            }
        }

        policy.insert(state.clone(), chosen);
    }

    policy
}

fn optimal_policy_cp6(cp: &CP6) -> DeterministicPolicy<CP6State, CP6Action> {
    let mut policy = HashMap::new();

    let pair12 = Product::new(PathAction::Next, PathAction::Next);
    let triple123 = Product::new(pair12, PathAction::Next);
    let four1234 = Product::new(triple123, PathAction::Next);
    let five12345 = Product::new(four1234, PathAction::Next);
    let target = Product::new(five12345, PathAction::Next);

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

fn run_trial6<M>(
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
    println!("6D path-product: Box vs Cartesian");

    let size = 3usize;
    let num_runs = 10usize;
    let num_episodes = 3000usize;

    let p1 = make_path_world(size);
    let p2 = make_path_world(size);
    let p3 = make_path_world(size);
    let p4 = make_path_world(size);
    let p5 = make_path_world(size);
    let p6 = make_path_world(size);

    let bp_temp1 = BoxProduct::new(p1, p2);
    let bp_temp2 = BoxProduct::new(bp_temp1, p3);
    let bp_temp3 = BoxProduct::new(bp_temp2, p4);
    let bp_temp4 = BoxProduct::new(bp_temp3, p5);
    let bp: BP6 = BoxProduct::new(bp_temp4, p6);

    let cp_temp1 = CartesianProduct::new(make_path_world(size), make_path_world(size));
    let cp_temp2 = CartesianProduct::new(cp_temp1, make_path_world(size));
    let cp_temp3 = CartesianProduct::new(cp_temp2, make_path_world(size));
    let cp_temp4 = CartesianProduct::new(cp_temp3, make_path_world(size));
    let cp: CP6 = CartesianProduct::new(cp_temp4, make_path_world(size));

    analyze_action_space(&bp, "BP");
    analyze_action_space(&cp, "CP");

    let opt_bp = optimal_policy_bp6(&bp);
    let opt_cp = optimal_policy_cp6(&cp);

    println!();
    println!(
        "Running {} trials ({} episodes each)...",
        num_runs, num_episodes
    );

    let mut bp_dists = Vec::with_capacity(num_runs);
    let mut cp_dists = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        bp_dists.push(run_trial6(&bp, &opt_bp, num_episodes, size * 7));
        cp_dists.push(run_trial6(&cp, &opt_cp, num_episodes, size * 7));
    }

    let bp_mean: f64 = bp_dists.iter().copied().sum::<f64>() / bp_dists.len() as f64;
    let cp_mean: f64 = cp_dists.iter().copied().sum::<f64>() / cp_dists.len() as f64;

    println!("\nResults:");
    println!("  BP policy distance: {:.4}", bp_mean);
    println!("  CP policy distance: {:.4}", cp_mean);
    println!("  Ratio (CP/BP): {:.2}x", cp_mean / bp_mean);
}
