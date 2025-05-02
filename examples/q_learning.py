from ctmdp.mdp import MDP
from .mdp import mdp_desc
from ctmdp.q_learning import q_learning

# Create an MDP instance using your provided description.
mdp_instance = MDP(mdp_desc)

# Run Q-learning for a specified number of episodes.
Q_table = q_learning(
    mdp_instance, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=50
)

# Print the resulting Q-table for each state.
for state, actions in Q_table.items():
    print(f"State: {state}")
    for action, value in actions.items():
        print(f"  Action: {action}, Q-value: {value:.3f}")
