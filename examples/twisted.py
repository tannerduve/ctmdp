from .mdp import mdp_desc
from ctmdp.mdp import MDP
from ctmdp.twisted import Automaton, TwistedMDP


# Example labeling function for testing (you would define this based on your scenario):
def example_label_func(state_label):
    x, y = state_label
    if x == 1 and y == 1:
        return "goal"
    elif x == 0 and y == 0:
        return "start"
    else:
        return "safe"


# Example Automaton for testing:
aut_states = ["q0", "q_goal"]
aut_alphabet = ["start", "safe", "goal"]
aut_transitions = {
    ("q0", "start"): "q0",
    ("q0", "safe"): "q0",
    ("q0", "goal"): "q_goal",
    ("q_goal", "start"): "q_goal",
    ("q_goal", "safe"): "q_goal",
    ("q_goal", "goal"): "q_goal",
}
aut_initial = "q0"
automaton = Automaton(
    aut_states, aut_alphabet, aut_transitions, aut_initial, accepting=["q_goal"]
)

# Example usage:
base_mdp = MDP(mdp_desc)
twisted_mdp = TwistedMDP(base_mdp, automaton, example_label_func)

# Inspecting twisted MDP structure:
print("\nTwisted MDP states and transitions:")
for label, state in twisted_mdp.states.items():
    print(f"State {label}:")
    for action_label, action in state.actions.items():
        print(
            f"  Action {action_label}: transitions {action.measure} with reward {action.reward}"
        )
