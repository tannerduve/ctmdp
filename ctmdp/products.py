from ctmdp.mdp import MDP
from functools import reduce, partial


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


def _box_product(M1, M2, labels=["M1", "M2"]):
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


def _cartesian_product(M1, M2, sep="-"):
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


# Relabeling function
def _append(args):
    l = args[0]
    e = args[1]
    return tuple([*l, e])


def _relabel(product):
    "Adds relabeling to a product operation."

    def f(*args, **kwargs):
        mdp = product(*args, **kwargs)
        mdp.relabel_all_states(_append)
        return mdp

    return f


def cartesian_product(*mdps, sep="-"):
    match len(mdps):
        case 1:
            return mdps[0]
        case 2:
            return _cartesian_product(*mdps, sep=sep)
        case _:
            tail = _cartesian_product(*mdps[-2::], sep=sep)
            body = mdps[:-2]
            return reduce(_relabel(_cartesian_product), body, tail)


def _n_ary_product(product):
    "Make a binary operation into a n-ary operation."

    def n_ary_product(*mdps, **kwargs):
        _product = partial(product, **kwargs)
        match len(mdps):
            case 1:
                return mdps[0]
            case 2:
                return _product(*mdps)
            case _:
                tail = _product(*mdps[-2::])
                body = mdps[:-2]
                return reduce(_relabel(_product), body, tail)

    return n_ary_product


cartesian_product = _n_ary_product(_cartesian_product)
box_product = _n_ary_product(_box_product)
