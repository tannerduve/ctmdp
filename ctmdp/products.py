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
            {
                switch_states((s.label, s_target.label)): w
                for s, w in action.measure.items()
            },
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


def product_measure(a1_measure, a2_measure):
    return {
        (s1.label, s2.label): (w1 + w2) / 2
        for s1, w1 in a1_measure.items()
        for s2, w2 in a2_measure.items()
    }


def cartesian_product(M1, M2, sep="-"):
    desc = {
        (s1.label, s2.label): {
            f"{a1.label}{sep}{a2.label}": product_measure(a1.measure, a2.measure)
            for a1 in s1.actions.values()
            for a2 in s2.actions.values()
        }
        for s1 in M1.states.values()
        for s2 in M2.states.values()
    }
    return MDP(desc)
