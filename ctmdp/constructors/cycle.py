from ctmdp.mdp import MDP


def cycle_mdp_actions(i):
    return {"prev": i - 1, "next": i + 1}


def cycle_mdp_desc(n):
    return {i: cycle_mdp_actions(i) for i in range(n)}


def cycle_mdp(n):
    return MDP(cycle_mdp_desc(n))
