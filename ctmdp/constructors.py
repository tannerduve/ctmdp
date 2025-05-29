from ctmdp.mdp import MDP


def cycle_mdp_actions(i):
    return {"prev": i - 1, "next": i + 1}


def cycle_mdp_desc(n):
    return {i: cycle_mdp_actions(i) for i in range(n)}


def cycle_mdp(n):
    return MDP(cycle_mdp_desc(n))


def path_mdp_actions(i, n):
    actions = cycle_mdp_actions(i)
    if i == 0:
        actions.pop("prev")
    if i == n - 1:
        actions.pop("next")
    return actions


def path_mdp_desc(n):
    return {i: path_mdp_actions(i, n) for i in range(n)}


def path_mdp(n):
    return MDP(path_mdp_desc(n))


def reward_action_label(label, reward):
    def reward_function(action):
        if action.label == label:
            return reward
        else:
            return 0

    return reward_function


def punish_non_action_label(label, punishment):
    def reward_function(action):
        if action.label != label:
            return -punishment
        else:
            return 0

    return reward_function


def reward_reaching(state, reward):
    def reward_function(action):
        if state in action.reachable_states:
            return reward
        else:
            return 0

    return reward_function
