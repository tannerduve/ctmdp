from ctmdp.mdp import MDP
from ctmdp.constructors.cycle import cycle_mdp_actions


def path_mdp_actions(i, n):
    actions = cycle_mdp_actions(i, n)
    if i == 0:
        actions.pop("prev")
    if i == n - 1:
        actions.pop("next")
    return actions


def path_mdp_desc(n):
    return {i: path_mdp_actions(i, n) for i in range(n)}


def path_mdp(n):
    return MDP(path_mdp_desc(n))


def optimal_policy(n):
    weights = {"next": 1, "prev": 0}
    desc = {
        i: {action: weights[action] for action in path_mdp_actions(i, n)}
        for i in range(n)
    }
    return desc
