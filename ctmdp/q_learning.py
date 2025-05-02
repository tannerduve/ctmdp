import random


def init_q(mdp):
    # Initialize Q table all zeros
    Q = {}
    for s_label, state_obj in mdp.states.items():
        Q[s_label] = {}
        for a_label, _ in state_obj.actions.items():
            Q[s_label][a_label] = 0.0  # Initialize Q-value to 0
    return Q


def e_greedy(Q, state_label, mdp, epsilon=0.1):
    """Return an action_label using an epsilon-greedy strategy:
    choose best action greedily but with epsilon probability of choosing randomly"""
    if random.random() < epsilon:
        # Choose random action
        return random.choice(list(mdp.states[state_label].actions.keys()))
    else:
        # Choose best action w.r.t. Q
        best_a = max(Q[state_label], key=lambda a: Q[state_label].get(a))  # argmax
        return best_a


# The agent updates Q-values using the formula:

# \[
# Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma Q(S', A') - Q(S, A) \right)
# \]

# Where:

# - **S** is the current state.
# - **A** is the action taken by the agent.
# - **S’** is the next state the agent moves to.
# - **A’** is the best next action in state **S’**.
# - **R** is the reward received for taking action **A** in state **S**.
# - **γ (Gamma)** is the **discount factor**, which balances immediate rewards with future rewards.
# - **α (Alpha)** is the **learning rate**, determining how much new information affects the old Q-values.


def q_update(Q, s, a, s_, r, alpha=0.1, gamma=0.9):
    """Update Q-value for state-action pair using the Q-learning update rule"""
    a_ = max(Q[s_], key=lambda a: Q[s_].get(a))  # argmax
    Q[s][a] += alpha * (r + gamma * Q[s_][a_] - Q[s][a])
    return Q


def q_learning(mdp, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=50):
    # 1. Initialize Q-table
    Q = init_q(mdp)

    for _ in range(episodes):
        # 2. Start state - pick something or random
        s_label = random.choice(list(mdp.states.keys()))
        for _ in range(max_steps):
            a_label = e_greedy(Q, s_label, mdp, epsilon)
            action = mdp.states[s_label].actions[a_label]
            next_state = action.transition()
            next_state_label = next_state.label
            reward = action.reward

            Q = q_update(Q, s_label, a_label, next_state_label, reward, alpha, gamma)
            s_label = next_state_label

    return Q
