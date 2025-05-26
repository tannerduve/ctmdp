from ctmdp.mdp import MDP


def _bp_action_data(states, labels, index):
    def switch_states(sts):
        if index == 0:
            return sts
        if index == 1:
            return sts[::-1]

    s_source = states[index]
    s_target = states[(index + 1) % 2]

    def labels_postfix(i):
        label = labels[i]
        if label:
            return f"-{label}"
        else:
            return ""

    return {
        f"{action.label}{labels_postfix(index)}": (
            {switch_states((k, s_target.label)): v for k, v in action.measure.items()},
            action.reward,
        )
        for action in s_source.actions.values()
    }


def box_product(M1, M2, labels=["M1", "M2"]):
    desc = {
        (s1.label, s2.label): _bp_action_data((s1, s2), labels, 0)
        | _bp_action_data((s1, s2), labels, 1)
        for s1 in M1.states.values()
        for s2 in M2.states.values()
    }
    return MDP(desc)
