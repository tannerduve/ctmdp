use std::collections::{HashMap, HashSet};

use ctmdp_rust::error::Error;
use ctmdp_rust::gridworld::GridworldWithGoals;
use ctmdp_rust::mdp::MDP;
use ctmdp_rust::products::{BoxAction, BoxProduct, CartesianProduct, Product};
use ctmdp_rust::q_learning::q_learning;
use madepro::environments::gridworld::{Cell, Gridworld, GridworldAction, GridworldState};
use madepro::models::{ActionValue, Config};

type DeterministicPolicy<S, A> = HashMap<S, A>;

struct WeightedGridworld {
    inner: GridworldWithGoals,
    goal_bonus: f64,
    step_scale: f64,
}

impl WeightedGridworld {
    fn new(inner: GridworldWithGoals, goal_bonus: f64, step_scale: f64) -> Self {
        Self {
            inner,
            goal_bonus,
            step_scale,
        }
    }
}

impl MDP for WeightedGridworld {
    type State = GridworldState;
    type Action = GridworldAction;

    fn all_states(&self) -> &madepro::models::Sampler<Self::State> {
        self.inner.all_states()
    }

    fn actions_at(&self, state: &Self::State) -> Vec<Self::Action> {
        self.inner.actions_at(state)
    }

    fn is_final_state(&self, state: &Self::State) -> bool {
        self.inner.is_final_state(state)
    }

    fn stochastic_transition(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Result<(ctmdp_rust::measure::Measure<Self::State>, f64), Error> {
        let (measure, reward) = self.inner.stochastic_transition(state, action)?;
        let mut adjusted = reward * self.step_scale;
        let hit_goal = measure
            .dist()
            .keys()
            .any(|next_state| self.inner.is_goal(next_state));
        if hit_goal {
            adjusted += self.goal_bonus;
        }
        Ok((measure, adjusted))
    }
}

fn grid_actions() -> Vec<GridworldAction> {
    vec![
        GridworldAction::Up,
        GridworldAction::Down,
        GridworldAction::Left,
        GridworldAction::Right,
    ]
}

fn grid_states(
    rows: usize,
    cols: usize,
    walls: &HashSet<(usize, usize)>,
) -> (Vec<GridworldState>, HashMap<GridworldState, (usize, usize)>) {
    let mut states = Vec::new();
    let mut coords = HashMap::new();
    for i in 0..rows {
        for j in 0..cols {
            if walls.contains(&(i, j)) {
                continue;
            }
            let state = GridworldState::new(i, j);
            coords.insert(state.clone(), (i, j));
            states.push(state);
        }
    }
    (states, coords)
}

fn grid_cells(
    rows: usize,
    cols: usize,
    walls: &HashSet<(usize, usize)>,
    goal: (usize, usize),
) -> Vec<Vec<Cell>> {
    let mut grid = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let cell = if walls.contains(&(i, j)) {
                Cell::Wall
            } else if (i, j) == goal {
                Cell::End
            } else {
                Cell::Air
            };
            row.push(cell);
        }
        grid.push(row);
    }
    grid
}

fn grid_cells_no_goal(
    rows: usize,
    cols: usize,
    walls: &HashSet<(usize, usize)>,
) -> Vec<Vec<Cell>> {
    let mut grid = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let cell = if walls.contains(&(i, j)) {
                Cell::Wall
            } else {
                Cell::Air
            };
            row.push(cell);
        }
        grid.push(row);
    }
    grid
}

fn build_component(
    size: (usize, usize),
    walls: &[(usize, usize)],
    goal: (usize, usize),
    goal_bonus: f64,
    step_scale: f64,
) -> (WeightedGridworld, HashMap<GridworldState, (usize, usize)>) {
    let wall_set: HashSet<(usize, usize)> = walls.iter().copied().collect();
    let (states, coords) = grid_states(size.0, size.1, &wall_set);
    let actions = grid_actions();
    let cells = grid_cells(size.0, size.1, &wall_set, goal);
    let grid = Gridworld::new(cells, states.clone(), actions);
    let goal_state = GridworldState::new(goal.0, goal.1);
    let gw = GridworldWithGoals::new(grid, vec![goal_state]);
    (WeightedGridworld::new(gw, goal_bonus, step_scale), coords)
}

fn build_penalty_component(
    size: (usize, usize),
    walls: &[(usize, usize)],
    step_scale: f64,
) -> WeightedGridworld {
    let wall_set: HashSet<(usize, usize)> = walls.iter().copied().collect();
    let (states, _coords) = grid_states(size.0, size.1, &wall_set);
    let actions = grid_actions();
    let cells = grid_cells_no_goal(size.0, size.1, &wall_set);
    let grid = Gridworld::new(cells, states, actions);
    let gw = GridworldWithGoals::new(grid, Vec::new());
    WeightedGridworld::new(gw, 0.0, step_scale)
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

fn policy_distance<S, A>(
    learned: &DeterministicPolicy<S, A>,
    optimal: &DeterministicPolicy<S, A>,
) -> f64
where
    S: Eq + std::hash::Hash,
    A: Eq,
{
    let mut total = 0.0;
    let mut count = 0.0;
    for (state, opt_action) in optimal {
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

fn manhattan(pos: (usize, usize), goal: (usize, usize)) -> usize {
    pos.0.abs_diff(goal.0) + pos.1.abs_diff(goal.1)
}

fn action_toward(pos: (usize, usize), goal: (usize, usize)) -> GridworldAction {
    if pos.0 < goal.0 {
        GridworldAction::Down
    } else if pos.0 > goal.0 {
        GridworldAction::Up
    } else if pos.1 < goal.1 {
        GridworldAction::Right
    } else if pos.1 > goal.1 {
        GridworldAction::Left
    } else {
        GridworldAction::Up
    }
}

fn optimal_policy_bp(
    bp: &BoxProduct<WeightedGridworld, WeightedGridworld>,
    coords_a: &HashMap<GridworldState, (usize, usize)>,
    coords_b: &HashMap<GridworldState, (usize, usize)>,
    goal_a: (usize, usize),
    goal_b: (usize, usize),
    weights: (f64, f64),
) -> DeterministicPolicy<
    Product<GridworldState, GridworldState>,
    BoxAction<GridworldAction, GridworldAction>,
> {
    let mut policy = HashMap::new();
    for state in bp.all_states().iter() {
        let pos_a = coords_a
            .get(state.first())
            .copied()
            .unwrap_or((0, 0));
        let pos_b = coords_b
            .get(state.second())
            .copied()
            .unwrap_or((0, 0));
        let dist_a = manhattan(pos_a, goal_a);
        let dist_b = manhattan(pos_b, goal_b);
        let action = if dist_a > 0 && (weights.0 * dist_a as f64 >= weights.1 * dist_b as f64 || dist_b == 0)
        {
            BoxAction::Left(action_toward(pos_a, goal_a))
        } else if dist_b > 0 {
            BoxAction::Right(action_toward(pos_b, goal_b))
        } else {
            BoxAction::Left(GridworldAction::Up)
        };
        policy.insert(state.clone(), action);
    }
    policy
}

fn optimal_policy_cp(
    cp: &CartesianProduct<WeightedGridworld, WeightedGridworld>,
    coords_a: &HashMap<GridworldState, (usize, usize)>,
    coords_b: &HashMap<GridworldState, (usize, usize)>,
    goal_a: (usize, usize),
    goal_b: (usize, usize),
) -> DeterministicPolicy<
    Product<GridworldState, GridworldState>,
    Product<GridworldAction, GridworldAction>,
> {
    let mut policy = HashMap::new();
    for state in cp.all_states().iter() {
        let pos_a = coords_a
            .get(state.first())
            .copied()
            .unwrap_or((0, 0));
        let pos_b = coords_b
            .get(state.second())
            .copied()
            .unwrap_or((0, 0));
        let act_a = action_toward(pos_a, goal_a);
        let act_b = action_toward(pos_b, goal_b);
        policy.insert(state.clone(), Product::new(act_a, act_b));
    }
    policy
}

fn evaluate_goal_hits<M, F>(
    mdp: &M,
    policy: &F,
    coords_a: &HashMap<GridworldState, (usize, usize)>,
    coords_b: &HashMap<GridworldState, (usize, usize)>,
    goal_a: (usize, usize),
    goal_b: (usize, usize),
    episodes: usize,
    max_steps: usize,
) -> Result<(f64, f64, f64), Error>
where
    M: MDP<State = Product<GridworldState, GridworldState>>,
    M::Action: Clone,
    F: Fn(&M::State) -> M::Action,
{
    let mut hit_a = 0.0;
    let mut hit_b = 0.0;
    let mut hit_both = 0.0;
    for _ in 0..episodes {
        let mut state = mdp.all_states().get_random().clone();
        let mut reached_a = false;
        let mut reached_b = false;
        for _ in 0..max_steps {
            let action = policy(&state);
            let (measure, _) = mdp.stochastic_transition(&state, &action)?;
            let next_state = measure
                .sample()
                .cloned()
                .unwrap_or_else(|| state.clone());
            if let Some(pos) = coords_a.get(next_state.first()) {
                if *pos == goal_a {
                    reached_a = true;
                }
            }
            if let Some(pos) = coords_b.get(next_state.second()) {
                if *pos == goal_b {
                    reached_b = true;
                }
            }
            state = next_state;
            if mdp.is_final_state(&state) {
                break;
            }
        }
        if reached_a {
            hit_a += 1.0;
        }
        if reached_b {
            hit_b += 1.0;
        }
        if reached_a && reached_b {
            hit_both += 1.0;
        }
    }
    let n = episodes as f64;
    Ok((hit_a / n, hit_b / n, hit_both / n))
}

fn analyze_action_space_bp_cp(
    bp: &BoxProduct<WeightedGridworld, WeightedGridworld>,
    cp: &CartesianProduct<WeightedGridworld, WeightedGridworld>,
) {
    let bp_states: Vec<_> = bp.all_states().iter().collect();
    let cp_states: Vec<_> = cp.all_states().iter().collect();
    let bp_actions: usize = bp_states.iter().map(|s| bp.actions_at(s).len()).sum();
    let cp_actions: usize = cp_states.iter().map(|s| cp.actions_at(s).len()).sum();
    println!(
        "  BP states: {}, avg actions/state: {:.1}",
        bp_states.len(),
        bp_actions as f64 / bp_states.len() as f64
    );
    println!(
        "  CP states: {}, avg actions/state: {:.1}",
        cp_states.len(),
        cp_actions as f64 / cp_states.len() as f64
    );
}

fn analyze_action_space_generic<M>(mdp: &M, label: &str)
where
    M: MDP,
{
    let states: Vec<_> = mdp.all_states().iter().collect();
    let total_actions: usize = states.iter().map(|s| mdp.actions_at(s).len()).sum();
    let avg = total_actions as f64 / states.len() as f64;
    println!("  {} states: {}", label, states.len());
    println!("  {} avg actions/state: {:.1}", label, avg);
}

fn evaluate_average_return<M>(
    mdp: &M,
    policy: &DeterministicPolicy<M::State, M::Action>,
    episodes: usize,
    max_steps: usize,
) -> Result<f64, Error>
where
    M: MDP,
    M::State: Clone,
    M::Action: Clone,
{
    let mut total_return = 0.0;
    for _ in 0..episodes {
        let mut state = mdp.all_states().get_random().clone();
        let mut episode_return = 0.0;
        for _ in 0..max_steps {
            let action = policy
                .get(&state)
                .cloned()
                .unwrap_or_else(|| mdp.actions_at(&state)[0].clone());
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

fn run_three_gridworld_experiment() -> Result<(), Error> {
    println!("\nThree-gridworld product with penalty dimensions");

    let size = (3, 3);
    let walls: Vec<(usize, usize)> = vec![];
    let goal_a = (0, 2);

    // Box product: high-value A, penalty-only B and C
    let (bp_a, _) = build_component(size, &walls, goal_a, 40.0, 1.0);
    let bp_b = build_penalty_component(size, &walls, 3.0);
    let bp_c = build_penalty_component(size, &walls, 3.0);
    let bp_temp = BoxProduct::new(bp_a, bp_b);
    let bp3 = BoxProduct::new(bp_temp, bp_c);

    // Cartesian product: same components, but rewards sum across all three
    let (cp_a, _) = build_component(size, &walls, goal_a, 40.0, 1.0);
    let cp_b = build_penalty_component(size, &walls, 3.0);
    let cp_c = build_penalty_component(size, &walls, 3.0);
    let cp_temp = CartesianProduct::new(cp_a, cp_b);
    let cp3 = CartesianProduct::new(cp_temp, cp_c);

    analyze_action_space_generic(&bp3, "BP3");
    analyze_action_space_generic(&cp3, "CP3");

    let mut config = Config::default();
    config.num_episodes = 3000;
    config.max_num_steps = 25;
    config.learning_rate = 0.1;
    config.discount_factor = 0.95;
    config.exploration_rate = 0.1;

    let q_bp3 = q_learning(&bp3, &config)?;
    let q_cp3 = q_learning(&cp3, &config)?;

    let policy_bp3 = greedy_policy(&bp3, &q_bp3);
    let policy_cp3 = greedy_policy(&cp3, &q_cp3);

    let eval_runs = 200;
    let bp3_return = evaluate_average_return(&bp3, &policy_bp3, eval_runs, 25)?;
    let cp3_return = evaluate_average_return(&cp3, &policy_cp3, eval_runs, 25)?;

    println!(
        "\nAverage episodic returns over {} evaluations (three-gridworld product):",
        eval_runs
    );
    println!("  BP3 average return: {:.2}", bp3_return);
    println!("  CP3 average return: {:.2}", cp3_return);

    Ok(())
}

fn main() -> Result<(), Error> {
    println!("Gridworld components with heterogeneous goals/rewards");

    let size = (3, 3);
    let walls: Vec<(usize, usize)> = vec![];
    let goal_a = (0, 2);
    let goal_b = (2, 0);

    let (bp_a, coords_a) = build_component(size, &walls, goal_a, 40.0, 1.0);
    let (bp_b, coords_b) = build_component(size, &walls, goal_b, 10.0, 1.0);
    let bp = BoxProduct::new(bp_a, bp_b);

    // Build fresh components for CP since constructors take ownership.
    let (cp_a, _) = build_component(size, &walls, goal_a, 40.0, 1.0);
    let (cp_b, _) = build_component(size, &walls, goal_b, 10.0, 1.0);
    let cp = CartesianProduct::new(cp_a, cp_b);

    analyze_action_space_bp_cp(&bp, &cp);

    let mut config = Config::default();
    config.num_episodes = 2000;
    config.max_num_steps = 20;
    config.learning_rate = 0.1;
    config.discount_factor = 0.95;
    config.exploration_rate = 0.1;

    let q_bp = q_learning(&bp, &config)?;
    let q_cp = q_learning(&cp, &config)?;

    let learned_bp = greedy_policy(&bp, &q_bp);
    let learned_cp = greedy_policy(&cp, &q_cp);

    let opt_bp = optimal_policy_bp(&bp, &coords_a, &coords_b, goal_a, goal_b, (1.0, 0.5));
    let opt_cp = optimal_policy_cp(&cp, &coords_a, &coords_b, goal_a, goal_b);

    let bp_dist = policy_distance(&learned_bp, &opt_bp);
    let cp_dist = policy_distance(&learned_cp, &opt_cp);

    println!("\nPolicy quality (L1/Hamming distance to heuristic optimum):");
    println!("  BP distance: {:.4}", bp_dist);
    println!("  CP distance: {:.4}", cp_dist);

    let eval_runs = 200;
    let bp_hits = evaluate_goal_hits(
        &bp,
        &|state| learned_bp
            .get(state)
            .cloned()
            .unwrap_or_else(|| BoxAction::Left(GridworldAction::Up)),
        &coords_a,
        &coords_b,
        goal_a,
        goal_b,
        eval_runs,
        20,
    )?;
    let cp_hits = evaluate_goal_hits(
        &cp,
        &|state| learned_cp
            .get(state)
            .cloned()
            .unwrap_or_else(|| Product::new(GridworldAction::Up, GridworldAction::Up)),
        &coords_a,
        &coords_b,
        goal_a,
        goal_b,
        eval_runs,
        20,
    )?;

    println!("\nGoal reach frequency over {eval_runs} evaluation rollouts:");
    println!(
        "  BP: goal A {:.1}% | goal B {:.1}% | both {:.1}%",
        bp_hits.0 * 100.0,
        bp_hits.1 * 100.0,
        bp_hits.2 * 100.0
    );
    println!(
        "  CP: goal A {:.1}% | goal B {:.1}% | both {:.1}%",
        cp_hits.0 * 100.0,
        cp_hits.1 * 100.0,
        cp_hits.2 * 100.0
    );

    run_three_gridworld_experiment()
}
