from .mdp import mdp_desc
from ctmdp.quotient import build_quotient_mdp
from ctmdp.mdp import MDP

mdp = MDP(mdp_desc)
quot_mdp, f, g = build_quotient_mdp(mdp)

print("\n========== Canonical State Mapping ==========")
for s, b in sorted(f.items()):
    print(f"State {s} → Block {b}")

print("\n========== Quotient MDP ==========")
for block_id in sorted(quot_mdp.states):
    print(f"\n--- Block {block_id} ---")
    state = quot_mdp.states[block_id]
    for action in sorted(state.actions.values(), key=lambda a: a.label):
        transitions = ", ".join(
            f"Block {target}: {prob:.2f}" for target, prob in action.measure.items()
        )
        print(f"  Action '{action.label}':")
        print(f"    Reward: {action.reward}")
        print(f"    Transitions: {transitions}")
