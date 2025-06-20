from ctmdp.mdp import Policy
from ctmdp.products import box_product, cartesian_product, _append
from ctmdp.constructors.path import path_mdp
import ctmdp.constructors.path as path
from ctmdp.rewards import (
    reward_action_label,
    penalize_non_action_label,
    reward_reaching,
)
from ctmdp.q_learning import q_learning_until

path1 = path_mdp(10)
path2 = path_mdp(15)

grid = box_product(path1, path2)
grid.relabel_actions(
    [("prev-M1", "left"), ("next-M1", "right"), ("prev-M2", "down"), ("next-M2", "up")]
)

path3 = path_mdp(20)
cube = box_product(grid, path3, labels=("", "M3"))
cube.relabel_actions([("prev-M3", "forward"), ("next-M3", "backward")])


cube.relabel_all_states(_append)


path1.set_rewards(reward_action_label("next", 1))
path1.set_rewards(penalize_non_action_label("next", 2))
path1.add_rewards(reward_reaching(path1.last_state, 10))

optimal_policy = Policy(path1, path.optimal_policy(10))
policy, episodes = q_learning_until(
    optimal_policy,
    path1,
    episodes=100,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    max_steps=20,
)

# cartesian product
cp = cartesian_product(path1, path1, path1)
bp = box_product(path1, path1, path1)


# optimal policy: diagaonal
def diagonal_policy(ncube):
    def diag(a_label):
        components = a_label.split("-")
        return int(all(c == "next" for c in components))

    desc = {
        s.label: {a.label: diag(a.label) for a in s.actions.values()}
        for s in ncube.states.values()
    }
    return Policy(ncube, desc)


optimal_policy = diagonal_policy(cp)


# rewards
def diag_rewards(action):
    components = action.label.split("-")

    def score(next_prev):
        match next_prev:
            case "next":
                return 1
            case "prev":
                return -2

    return sum(score(c) for c in components)


cp.set_rewards(diag_rewards)

# training
policy, episodes = q_learning_until(
    optimal_policy,
    cp,
    episodes=1000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    max_steps=20,
)

# policy on product from components
# refine policy products as a measure for generalization
