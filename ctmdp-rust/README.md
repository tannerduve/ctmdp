## ctmdp-rust

This crate is the Rust counterpart to the Python `ctmdp` experiments. The goal is to run Q-learning on small MDPs (gridworlds, chains, and their products) with proper types and explicit probability measures.

We build on the community crate **`madepro`** (`madepro::models` and `madepro::environments::gridworld`) for base traits (`State`, `Action`, `Sampler`, `MDP`) and a reference `Gridworld` implementation.

### What’s here

- `mdp.rs`: local `MDP` trait for *stochastic* environments (`stochastic_transition` returns a `Measure<State>` + reward).
- `measure.rs`: `Probability` (checked `[0,1]` float) and `Measure<T>` (discrete distribution over states, plus a `product` constructor for independent components).
- `gridworld.rs`: adapter from `madepro`’s `Gridworld` to our `MDP` trait, plus `GridworldWithGoals` (adds explicit goal states).
- `pathmdp.rs`: 1D chain MDP (`PathWorld`) with `Next`/`Prev` actions and rewards tuned to match the Python chain experiments.
- `products.rs`: (IN PROGRESS) Box and Cartesian products of MDPs (`BoxProduct`, `CartesianProduct`, custom product state/action types).
- `q_learning.rs`: (TODO) stochastic Q-learning implementation.

### Status

- Core types (`MDP`, `Measure`, `Gridworld` adapter, `PathWorld`) are in place.
- Product MDPs are in progress and Q-learning is todo.
- `Cargo.toml` defines experiment binaries (`compare_products`, `compare_chains`, `compare_3d`, `compare_4d`) that will mirror the Python experiments once Q-learning and products are wired up.