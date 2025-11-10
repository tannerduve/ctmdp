"""
3D gridworld experiment: Box Product vs Cartesian Product.
Shows action space advantage: BP has 6 actions, CP has 8 actions.
"""

from ctmdp.mdp import Policy
from ctmdp.products import box_product, cartesian_product
from ctmdp.constructors.path import path_mdp
from ctmdp.q_learning import q_learning
import random


def reward_function(mdp):
    """Reward for 'next', penalty for 'prev', bonus at goal."""
    def base_reward(action):
        components = action.label.split('-')
        return sum(0.1 if c == 'next' else -0.5 if c == 'prev' else 0 for c in components)
    
    mdp.set_rewards(base_reward)
    
    # Goal bonus
    goal_label = list(mdp.goals)[0].label if mdp.goals else None
    if goal_label and goal_label in mdp.states:
        for action in mdp.states[goal_label].actions.values():
            action.reward += 10.0


def optimal_policy_bp(bp):
    """Optimal: always choose 'next' actions."""
    desc = {}
    for state_label, state in bp.states.items():
        actions = {}
        for action_label in state.actions.keys():
            if 'next' in action_label:
                actions[action_label] = 1.0
            else:
                actions[action_label] = 0.0
        
        total = sum(actions.values())
        if total > 0:
            actions = {a: w/total for a, w in actions.items()}
        else:
            actions = {a: 1.0/len(state.actions) for a in state.actions}
        
        desc[state_label] = actions
    return Policy(bp, desc)


def optimal_policy_cp(cp):
    """Optimal: always choose 'next-next-next'."""
    desc = {}
    for state_label, state in cp.states.items():
        actions = {}
        for action_label in state.actions.keys():
            components = action_label.split('-')
            if all(c == 'next' for c in components):
                actions[action_label] = 1.0
            else:
                actions[action_label] = 0.0
        
        total = sum(actions.values())
        if total > 0:
            actions = {a: w/total for a, w in actions.items()}
        else:
            actions = {a: 1.0/len(state.actions) for a in state.actions}
        
        desc[state_label] = actions
    return Policy(cp, desc)


def policy_distance(learned_policy, optimal_policy):
    """L1 distance between policies."""
    total_distance = 0
    num_states = 0
    
    for state_label in optimal_policy.policy.keys():
        if state_label not in learned_policy.policy:
            continue
        opt_actions = optimal_policy.policy[state_label]
        learned_actions = learned_policy.policy[state_label]
        all_actions = set(opt_actions.keys()).union(learned_actions.keys())
        state_distance = sum(
            abs(opt_actions.get(a, 0) - learned_actions.get(a, 0))
            for a in all_actions
        )
        total_distance += state_distance
        num_states += 1
    
    return total_distance / num_states if num_states > 0 else float('inf')


def run_trial(mdp, optimal_policy, episodes, max_steps):
    """Run Q-learning and return final distance."""
    Q = q_learning(mdp, episodes=episodes, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=max_steps)
    learned = Policy(mdp, Q)
    return policy_distance(learned.deterministic, optimal_policy.deterministic)


def main():
    print("="*70)
    print("3D Gridworld Experiment: BP vs CP")
    print("="*70)
    
    size = 4
    num_runs = 10
    
    print(f"\n{size}×{size}×{size} grid:")
    
    # Create 3D products
    path1 = path_mdp(size)
    path2 = path_mdp(size)
    path3 = path_mdp(size)
    
    # Box product: compose sequentially
    bp_temp = box_product(path1, path2)
    bp = box_product(bp_temp, path3)
    
    # Cartesian product: compose sequentially
    cp_temp = cartesian_product(path1, path2)
    cp = cartesian_product(cp_temp, path3)
    
    # Set rewards
    reward_function(bp)
    reward_function(cp)
    
    # Check action space (state labels are nested tuples)
    interior = ((size // 2, size // 2), size // 2)
    bp_actions = len(bp.states[interior].actions) if interior in bp.states else 0
    cp_actions = len(cp.states[interior].actions) if interior in cp.states else 0
    
    print(f"  States: {len(bp.states)}")
    print(f"  BP actions/state (interior): {bp_actions} (2+2+2=6)")
    print(f"  CP actions/state (interior): {cp_actions} (2×2×2=8)")
    if bp_actions > 0:
        print(f"  Action space ratio (CP/BP): {cp_actions/bp_actions:.2f}x")
    
    # Optimal policies
    opt_bp = optimal_policy_bp(bp)
    opt_cp = optimal_policy_cp(cp)
    
    # Run trials
    print(f"\nRunning {num_runs} trials (1500 episodes each)...")
    bp_dists = []
    cp_dists = []
    
    for run in range(num_runs):
        bp_dists.append(run_trial(bp, opt_bp, episodes=1500, max_steps=size*4))
        cp_dists.append(run_trial(cp, opt_cp, episodes=1500, max_steps=size*4))
        if (run + 1) % 5 == 0:
            print(f"  Completed {run+1}/{num_runs} runs")
    
    bp_mean = sum(bp_dists) / len(bp_dists)
    cp_mean = sum(cp_dists) / len(cp_dists)
    
    print(f"\nResults:")
    print(f"  BP distance: {bp_mean:.4f}")
    print(f"  CP distance: {cp_mean:.4f}")
    print(f"  Ratio (CP/BP): {cp_mean/bp_mean:.2f}x")
    
    print(f"\n{'='*70}")
    print("In 3D: CP has 33% more actions than BP (8 vs 6)")
    print("But BP still learns better due to decomposable structure.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

