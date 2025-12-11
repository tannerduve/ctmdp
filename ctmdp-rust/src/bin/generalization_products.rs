use std::collections::HashMap;

use ctmdp_rust::error::Error;
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
    M::State: Clone + std::hash::Hash + Eq,
    M::Action: Clone + std::hash::Hash + Eq,
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

fn train_path_policy(length: usize, episodes: u32, max_steps: u32) -> DeterministicPolicy<PathState, PathAction> {
    let world = make_path_world(length);
    let mut config = Config::default();
    config.num_episodes = episodes;
    config.max_num_steps = max_steps;
    config.learning_rate = 0.1;
    config.discount_factor = 0.95;
    config.exploration_rate = 0.1;
    let q_values = q_learning(&world, &config).expect("component training failed");
    greedy_policy(&world, &q_values)
}

fn evaluate_policy<M, F>(
    mdp: &M,
    policy: &F,
    episodes: usize,
    max_steps: usize,
) -> Result<f64, Error>
where
    M: MDP,
    M::State: Clone,
    M::Action: Clone,
    F: Fn(&M::State) -> M::Action,
{
    let mut total_return = 0.0;
    for _ in 0..episodes {
        let mut state = mdp.all_states().get_random().clone();
        let mut episode_return = 0.0;
        for _ in 0..max_steps {
            let action = policy(&state);
            let (measure, reward) = mdp.stochastic_transition(&state, &action)?;
            episode_return += reward;
            let next_state = measure
                .sample()
                .cloned()
                .unwrap_or_else(|| state.clone());
            state = next_state;
            if mdp.is_final_state(&state) {
                break;
            }
        }
        total_return += episode_return;
    }
    Ok(total_return / episodes as f64)
}

fn composed_box_policy<'a>(
    policy_a: &'a DeterministicPolicy<PathState, PathAction>,
    policy_b: &'a DeterministicPolicy<PathState, PathAction>,
    length: usize,
) -> impl Fn(&Product<PathState, PathState>) -> BoxAction<PathAction, PathAction> + 'a {
    move |state| {
        let target_index = length - 1;
        if state.first().index() != target_index {
            let act = policy_a
                .get(state.first())
                .cloned()
                .unwrap_or(PathAction::Next);
            BoxAction::Left(act)
        } else {
            let act = policy_b
                .get(state.second())
                .cloned()
                .unwrap_or(PathAction::Next);
            BoxAction::Right(act)
        }
    }
}

fn composed_cartesian_policy<'a>(
    policy_a: &'a DeterministicPolicy<PathState, PathAction>,
    policy_b: &'a DeterministicPolicy<PathState, PathAction>,
) -> impl Fn(&Product<PathState, PathState>) -> Product<PathAction, PathAction> + 'a {
    move |state| {
        let act_a = policy_a
            .get(state.first())
            .cloned()
            .unwrap_or(PathAction::Next);
        let act_b = policy_b
            .get(state.second())
            .cloned()
            .unwrap_or(PathAction::Next);
        Product::new(act_a, act_b)
    }
}

fn learned_box_policy<'a>(
    policy: &'a DeterministicPolicy<
        Product<PathState, PathState>,
        BoxAction<PathAction, PathAction>,
    >,
) -> impl Fn(&Product<PathState, PathState>) -> BoxAction<PathAction, PathAction> + 'a {
    move |state| {
        policy
            .get(state)
            .cloned()
            .unwrap_or_else(|| BoxAction::Left(PathAction::Next))
    }
}

fn learned_cartesian_policy<'a>(
    policy: &'a DeterministicPolicy<
        Product<PathState, PathState>,
        Product<PathAction, PathAction>,
    >,
) -> impl Fn(&Product<PathState, PathState>) -> Product<PathAction, PathAction> + 'a {
    move |state| {
        policy
            .get(state)
            .cloned()
            .unwrap_or_else(|| Product::new(PathAction::Next, PathAction::Next))
    }
}

fn main() -> Result<(), Error> {
    println!("Generalization via component policies: Box vs Cartesian");

    let length = 10;
    let component_policy = train_path_policy(length, 800, 20);
    let component_policy_b = component_policy.clone();

    let bp = BoxProduct::new(make_path_world(length), make_path_world(length));
    let cp = CartesianProduct::new(make_path_world(length), make_path_world(length));

    let mut product_config = Config::default();
    product_config.num_episodes = 2000;
    product_config.max_num_steps = (length as u32) * 3;
    product_config.learning_rate = 0.1;
    product_config.discount_factor = 0.95;
    product_config.exploration_rate = 0.1;

    let q_bp = q_learning(&bp, &product_config)?;
    let q_cp = q_learning(&cp, &product_config)?;

    let learned_bp = greedy_policy(&bp, &q_bp);
    let learned_cp = greedy_policy(&cp, &q_cp);

    let eval_episodes = 200;
    let eval_steps = length * 4;

    let composed_bp_return = evaluate_policy(
        &bp,
        &composed_box_policy(&component_policy, &component_policy_b, length),
        eval_episodes,
        eval_steps,
    )?;
    let learned_bp_return = evaluate_policy(
        &bp,
        &learned_box_policy(&learned_bp),
        eval_episodes,
        eval_steps,
    )?;

    let composed_cp_return = evaluate_policy(
        &cp,
        &composed_cartesian_policy(&component_policy, &component_policy_b),
        eval_episodes,
        eval_steps,
    )?;
    let learned_cp_return = evaluate_policy(
        &cp,
        &learned_cartesian_policy(&learned_cp),
        eval_episodes,
        eval_steps,
    )?;

    println!("\nAverage episodic returns over {eval_episodes} evaluations:");
    println!("  BP composed policy (zero-shot): {:.2}", composed_bp_return);
    println!("  BP trained from scratch     : {:.2}", learned_bp_return);
    println!("  CP composed policy (zero-shot): {:.2}", composed_cp_return);
    println!("  CP trained from scratch       : {:.2}", learned_cp_return);

    Ok(())
}
