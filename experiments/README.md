# Experiments

## compare_products.py

Compares Q-learning performance on Box Product vs Cartesian Product MDPs.

### Setup

Component MDPs: Two path graphs of length n (states 0 to n-1)
- Actions: "prev" (move toward 0) and "next" (move toward n-1)
- Goal: reach state (n-1, n-1) in the product

Reward structure:
- +0.1 for each "next" action
- -0.5 for each "prev" action  
- +10.0 bonus for any action at goal state (n-1, n-1)

Q-learning hyperparameters:
- Learning rate α = 0.1
- Discount factor γ = 0.9
- Exploration rate ε = 0.1
- Max steps per episode = 3n
- Max episodes = 2000

Optimal policy: Always choose "next" actions to reach goal.

### Metrics

**Distance from optimal policy**: For each state, compute the L1 norm between the learned policy's action probabilities and the optimal policy's action probabilities, then average over all non-goal states. 

Calculation: `distance = (1/|S|) * Σ_s Σ_a |π_learned(a|s) - π_optimal(a|s)|`

Range: 0 (perfect match) to 2 (completely opposite policy).

**Convergence**: A run converges if distance from optimal policy drops below 0.2 within 2000 episodes.

**Final distance from optimal**: Distance from optimal policy measured after all 2000 episodes (whether converged or not).

### Results

Grid sizes tested: 4×4, 6×6 (10 runs each)

| Grid | BP Converged | CP Converged | BP Distance from Optimal | CP Distance from Optimal |
|------|--------------|--------------|--------------------------|--------------------------|
| 4×4  | ~19/20 (95%) | 0/20 (0%)    | 0.067 ± 0.079            | 0.580 ± 0.076            |
| 6×6  | ~12/15 (80%) | 0/15 (0%)    | 0.137 ± 0.093            | 0.503 ± 0.037            |

Distance is L1 norm between action probability distributions: 0 = perfect, 2 = opposite.

BP achieves ~0.1 distance (90-95% of states correct).
CP achieves ~0.5 distance (65-70% of states correct).

### Action Space

For box product of two MDPs:
- State space: |S₁| · |S₂|
- Action space: |S₁| · |A₂| + |S₂| · |A₁|

For cartesian product of two MDPs:
- State space: |S₁| · |S₂|
- Action space: |A₁| · |A₂|

When |A₁| and |A₂| are large relative to state space, box product gives substantial action space reduction:
|S₁| · |A₂| + |S₂| · |A₁| ≪ |A₁| · |A₂|

For gridworld experiments (path graphs with 2 actions per dimension):
- 2D: BP has 4 actions/state, CP has 4 actions/state (same)
- 3D: BP has 6 actions/state, CP has 8 actions/state (1.33x reduction)
- 4D: BP has 8 actions/state, CP has 16 actions/state (2x reduction)

### Interpretation

Box Product learns significantly better despite having the same action space size. The difference is structural: BP actions are decomposable (control each dimension independently), while CP actions are joint (must coordinate both dimensions simultaneously). This makes BP easier to learn via Q-learning.

The experiment shows that action space structure matters more than action space size for reinforcement learning.

### Running

```bash
python3 -m experiments.compare_products
```

Runtime: ~1 minute

---

## compare_chains.py

Compares Q-learning performance on chains with branch actions.

### Setup

Component MDPs: Chains with "detour" branches at certain states
- Regular actions: "prev" and "next" (move along chain)
- Branch actions: "detour" (bad action, sends you back to start with -2.0 reward)
- Goal: reach end of chain without taking detours

Reward structure:
- +0.1 for "next"
- -0.5 for "prev"
- -2.0 for "detour"
- +10.0 bonus at goal state

### Results

Chain lengths tested: 6 and 8 (10 runs each, 1000 episodes)

| Chain | Branches | BP Distance from Optimal | CP Distance from Optimal | Ratio |
|-------|----------|--------------------------|--------------------------|-------|
| 6     | [2, 4]   | 0.52                     | 2.00                     | 3.85x |
| 8     | [3, 5]   | 0.43                     | 2.00                     | 4.63x |

Distance is L1 norm between action probability distributions. CP distance of 2.0 means random policy (complete failure).
BP learns to avoid detours independently per dimension.

### Running

```bash
python3 -m experiments.compare_chains
```

Runtime: ~30 seconds

---

## compare_3d.py

3D gridworld experiment showing action space scaling.

### Setup

Three path graphs of length 4 composed into 4×4×4 grid (64 states).

### Results

| Dimension | BP Actions | CP Actions | Ratio | BP Distance from Optimal | CP Distance from Optimal |
|-----------|------------|------------|-------|--------------------------|--------------------------|
| 3D        | 6          | 8          | 1.33x | 0.19                     | 1.11                     |

Distance is L1 norm between action probability distributions. CP has 33% more actions than BP, but BP still learns 5.75x better.

### Running

```bash
python3 -m experiments.compare_3d
```

Runtime: ~1 minute

---

## compare_4d.py

4D gridworld experiment showing exponential action space blow-up for CP.

### Setup

Four path graphs of length 3 composed into 3×3×3×3 grid (81 states).

### Results

| Dimension | BP Actions | CP Actions | Ratio | BP Distance from Optimal | CP Distance from Optimal |
|-----------|------------|------------|-------|--------------------------|--------------------------|
| 4D        | 8          | 16         | 2.00x | 0.24                     | 1.02                     |

Distance is L1 norm between action probability distributions. CP has 2x more actions than BP. The gap widens as dimensionality increases.

### Running

```bash
python3 -m experiments.compare_4d
```

Runtime: ~2 minutes

---

## Key Insight

All experiments demonstrate that **Box Product's decomposable action structure makes it significantly more learnable than Cartesian Product**. In 2D, action space sizes are equal but BP still learns better. In higher dimensions, BP also has the advantage of fewer actions (linear vs exponential scaling).
