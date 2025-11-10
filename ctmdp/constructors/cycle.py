from ctmdp.mdp import MDP


def cycle_mdp_actions(i, n):
    return {"prev": (i - 1) % n, "next": (i + 1) % n}


def cycle_mdp_desc(n):
    return {i: cycle_mdp_actions(i, n) for i in range(n)}


def cycle_mdp(n):
    return MDP(cycle_mdp_desc(n))
