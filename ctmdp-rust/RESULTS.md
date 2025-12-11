# ctmdp-rust experiment results

## 3D path product (`compare_3d`)

- Task: three 1D chains combined into a 3D path with Next/Prev actions.
- Metric: policy distance to an "all Next" optimal policy.
- Action spaces:
  - BP actions/state: 6
  - CP actions/state: 8
- Results (10 runs, 1500 episodes each):
  - Size: 4
  - BP states: 64
  - BP avg actions/state: 6.0
  - CP states: 64
  - CP avg actions/state: 8.0
  - BP policy distance: 0.7109
  - CP policy distance: 0.7594
  - Ratio (CP/BP): 1.07x
- Takeaway: In 3D, BP learns a slightly better policy than CP despite a smaller action space.

## 4D path product (`compare_4d`)

- Task: four 1D chains combined into a 4D path with the same reward structure and metric as 3D.
- Action spaces:
  - BP actions/state: 8
  - CP actions/state: 16
- Results (10 runs, 2000 episodes each):
  - Size: 3
  - BP states: 81
  - BP avg actions/state: 8.0
  - CP states: 81
  - CP avg actions/state: 16.0
  - BP policy distance: 0.8235
  - CP policy distance: 0.9086
  - Ratio (CP/BP): 1.10x
- Takeaway: In 4D, BP's advantage over CP increases as the joint action space grows.

## 6D path product (`compare_6d`)

- Task: six 1D chains combined into a 6D path with the same reward structure and metric as 3D/4D.
- Action spaces:
  - BP actions/state: 12
  - CP actions/state: 64
- Results (10 runs, 3000 episodes each):
  - Size: 3
  - BP states: 729
  - BP avg actions/state: 12.0
  - CP states: 729
  - CP avg actions/state: 64.0
  - BP policy distance: 0.8610
  - CP policy distance: 0.9709
  - Ratio (CP/BP): 1.13x
- Takeaway: In 6D, BP clearly outperforms CP while using far fewer actions per state.

## Three-grid gridworld with penalty dimensions (`compare_gridworld_products`)

- Task: three 3x3 gridworlds; grid A has a goal, grids B and C have only step penalties.
- Control: BP acts in one grid per step; CP acts in all three each step.
- Metric: average episodic return after training.
- Action spaces:
  - BP3 actions/state: 12
  - CP3 actions/state: 64
- Results (200 evaluation episodes):
  - BP3 states: 729
  - CP3 states: 729
  - BP3 average return: 975.19
  - CP3 average return: 875.04
- Takeaway: BP3 focuses actions on the rewarding grid and mostly ignores the penalty grids, achieving higher return than CP3, which must pay costs in all dimensions every step.
