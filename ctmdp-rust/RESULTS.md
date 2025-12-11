ctmdp-rust experiment results

compare_3d (3D path product)
Task: 3D path product (three 1D chains) with Next/Prev actions; metric is policy distance to an “all Next” optimal policy. CP has 8 actions/state, BP has 6.
Results:
  Size: 4
  BP states: 64
  BP avg actions/state: 6.0
  CP states: 64
  CP avg actions/state: 8.0
  Trials: 10, episodes per trial: 1500
  BP policy distance: 0.7109
  CP policy distance: 0.7594
  Ratio (CP/BP): 1.07x
Takeaway: In 3D, BP learns a slightly better policy than CP despite a smaller action space.

compare_4d (4D path product)
Task: 4D path product (four 1D chains) with the same reward structure and metric as the 3D case; CP’s joint action space doubles relative to BP.
Results:
  Size: 3
  BP states: 81
  BP avg actions/state: 8.0
  CP states: 81
  CP avg actions/state: 16.0
  Trials: 10, episodes per trial: 2000
  BP policy distance: 0.8235
  CP policy distance: 0.9086
  Ratio (CP/BP): 1.10x
Takeaway: In 4D, BP’s advantage over CP increases, indicating better scaling in higher-dimensional products.

compare_6d (6D path product)
Task: 6D path product (six 1D chains) with the same reward structure and metric; CP’s action space grows much faster than BP’s (64 vs 12 actions per state).
Results:
  Size: 3
  BP states: 729
  BP avg actions/state: 12.0
  CP states: 729
  CP avg actions/state: 64.0
  Trials: 10, episodes per trial: 3000
  BP policy distance: 0.8610
  CP policy distance: 0.9709
  Ratio (CP/BP): 1.13x
Takeaway: In 6D, BP clearly outperforms CP while using far fewer actions per state, showing a strong scaling advantage.

compare_gridworld_products (gridworld)
Part 1: two-grid product with heterogeneous goals
  Task: two 3x3 gridworlds A and B, each with its own goal and reward scale; metric is policy distance to a heuristic directional policy and empirical goal-hit rates. Both components are useful to control.
  Results:
    BP states: 81, avg actions/state: 8.0
    CP states: 81, avg actions/state: 16.0
    Policy distance to heuristic:
      BP distance: 0.7531
      CP distance: 0.8519
    Goal reach frequency over 200 evaluations:
      BP: goal A 55.5% | goal B 47.5% | both 3.0%
      CP: goal A 67.5% | goal B 55.5% | both 23.0%
  Takeaway: With two meaningful tasks, CP is better at pushing both agents to their goals, while BP stays closer to a simple heuristic; this is not an async-control showcase.

Part 2: three-grid product with penalty dimensions
  Task: three 3x3 gridworlds, where A has a goal and B,C are penalty-only grids; BP can act in one grid per step, CP must act in all three; metric is average episodic return after training. This models a primary task (A) plus nuisance subsystems (B,C) where acting is always costly.
  Results:
    BP3 states: 729
    BP3 avg actions/state: 12.0
    CP3 states: 729
    CP3 avg actions/state: 64.0
    Average episodic returns over 200 evaluations:
      BP3 average return: 975.19
      CP3 average return: 875.04
  Takeaway: In this async-control scenario, BP3 clearly outperforms CP3 by acting mostly on the valuable grid and largely ignoring the penalty grids; CP3 is forced to pay costs in all dimensions every step.

generalization_products (component policy reuse on chains)
Task: learn a policy on a single 1D chain, then reuse it zero-shot on a 2D chain product (BP/CP) and compare to policies trained from scratch; metric is average episodic return. This probes how naturally component policies transfer to product MDPs.
Results (200 evaluation episodes):
  BP composed policy (zero-shot): 18.70
  BP trained from scratch: 160.36
  CP composed policy (zero-shot): 184.89
  CP trained from scratch: 172.12
Takeaway: The naive component-to-product composition favors CP in this setup; BP benefits from training directly on the product, so this experiment mainly serves as a compositionality check, not evidence for or against async control.
