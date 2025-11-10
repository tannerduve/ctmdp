"""
Experiment 3: Chains with branches.
Tests heterogeneous action spaces (some states have more actions).
"""

from ctmdp.mdp import MDP, Policy
from ctmdp.products import box_product, cartesian_product
from ctmdp.q_learning import q_learning
import random


def chain_with_branches(n, branch_states):
    """
    Chain of length n with optional detour branches at specified states.
    - Regular states: 'next', 'prev'
    - Branch states: 'next', 'prev', 'detour' (bad action, returns to start)
    """
    desc = {}
    for i in range(n):
        actions = {'prev': max(0, i-1), 'next': min(n-1, i+1)}
        if i in branch_states:
            actions['detour'] = 0  # Detour sends you back to start
        desc[i] = actions
    
    mdp = MDP(desc)
    
    # Set rewards
    for state in mdp.states.values():
        for action_label, action in state.actions.items():
            if action_label == 'next':
                action.reward = 0.1
            elif action_label == 'prev':
                action.reward = -0.5
            elif action_label == 'detour':
                action.reward = -2.0  # Bad!
    
    # Goal bonus
    goal_state = n - 1
    if goal_state in mdp.states:
        for action in mdp.states[goal_state].actions.values():
            action.reward += 10.0
    
    return mdp


def optimal_policy_bp(bp):
    """Optimal: choose 'next' actions, avoid detours."""
    desc = {}
    for state_label, state in bp.states.items():
        actions = {}
        for action_label in state.actions.keys():
            # Prefer 'next', avoid 'detour'
            if 'next' in action_label:
                actions[action_label] = 1.0
            else:
                actions[action_label] = 0.0
        
        total = sum(actions.values())
        if total > 0:
            actions = {a: w/total for a, w in actions.items()}
        else:
            # If no 'next' available (shouldn't happen), uniform
            actions = {a: 1.0/len(state.actions) for a in state.actions}
        
        desc[state_label] = actions
    return Policy(bp, desc)


def optimal_policy_cp(cp):
    """Optimal: choose 'next-next', avoid any detours."""
    desc = {}
    for state_label, state in cp.states.items():
        actions = {}
        for action_label in state.actions.keys():
            components = action_label.split('-')
            # Both should be 'next'
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
    print("Chain with Branches Experiment: BP vs CP")
    print("="*70)
    
    configs = [
        (6, [2, 4]),   # 6-length chain, branches at positions 2 and 4
        (8, [3, 5]),   # 8-length chain, branches at positions 3 and 5
    ]
    
    num_runs = 10
    
    for length, branches in configs:
        print(f"\nChain length {length}, branches at {branches}:")
        
        # Create chains
        chain1 = chain_with_branches(length, branches)
        chain2 = chain_with_branches(length, branches)
        
        bp = box_product(chain1, chain2)
        cp = cartesian_product(chain1, chain2)
        
        # Optimal policies
        opt_bp = optimal_policy_bp(bp)
        opt_cp = optimal_policy_cp(cp)
        
        # Count actions
        bp_avg_actions = sum(len(s.actions) for s in bp.states.values()) / len(bp.states)
        cp_avg_actions = sum(len(s.actions) for s in cp.states.values()) / len(cp.states)
        
        print(f"  Avg actions/state: BP={bp_avg_actions:.1f}, CP={cp_avg_actions:.1f}")
        
        # Run trials
        bp_dists = []
        cp_dists = []
        
        for _ in range(num_runs):
            bp_dists.append(run_trial(bp, opt_bp, episodes=1000, max_steps=length*3))
            cp_dists.append(run_trial(cp, opt_cp, episodes=1000, max_steps=length*3))
        
        bp_mean = sum(bp_dists) / len(bp_dists)
        cp_mean = sum(cp_dists) / len(cp_dists)
        
        print(f"  BP distance: {bp_mean:.4f}")
        print(f"  CP distance: {cp_mean:.4f}")
        print(f"  Ratio: {cp_mean/bp_mean:.2f}x")
    
    print(f"\n{'='*70}")
    print("Heterogeneous action spaces - some states have detour actions.")
    print("BP learns to avoid detours independently per dimension.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

