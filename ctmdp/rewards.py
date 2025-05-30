def reward_action_label(label, reward):
    def reward_function(action):
        if action.label == label:
            return reward
        else:
            return 0

    return reward_function


def penalize_non_action_label(label, penalty):
    def reward_function(action):
        if action.label != label:
            return -penalty
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
