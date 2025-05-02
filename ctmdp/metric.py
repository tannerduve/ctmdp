import ctmdp
import random


class MetricMDP(ctmdp.MDP):
    def __init__(self, description, distance_fn):
        """
        Initialize a Metric MDP with a distance function.
        """
        super().__init__(description)
        self.distance = distance_fn

    def state_distance(self, s1label, s2label):
        """
        Compute the distance between two states.
        """
        return self.distance(s1label, s2label)


class MetricMDPMorphism:
    def __init__(self, source, target, f, g, epsilon_fn, dR, W):
        self.source = source  # an MDP
        self.target = target  # an MDP
        self.f = f  # S1 -> S2
        self.g = g  # A1 -> A2
        self.epsilon = epsilon_fn  # A1 -> [0,1)
        self.dR = dR  # reward metric
        self.W = W  # Wasserstein-type metric on distributions

    def is_compatible(self):
        for a1 in self.source.all_actions():
            a2 = self.g(a1)
            R1 = self.source.get_reward(a1)
            R2 = self.target.get_reward(a2)
            T1 = self.source.get_transition(a1)
            T2 = self.target.get_transition(a2)
            pushed = self.pushforward(T1, self.f)
            err = self.dR(R1, R2) + self.W(pushed, T2)
            if err > self.epsilon(a1):
                return False
        return True


# Domain-specific generalization algorithm:
# Input: An MDP M, a set of MDPs {G_i}{i \in I}
# Output: A morphism M -> G_i for some i
# Algorithm:
# 1. For each G_i,
#   randomly initialize maps f, g
#   minimize the error function of the morphism
# 2. Select G \in {G_i}_{i \in I} and a pair (f,g) such that the error is minimized over all pairs (f', g') for all G_i
# 3. For each action a in M,
#    \epsilon(a) <- dR(R(a), R'(g(a))) + W(pushforward(T(a), f), T'(g(a)))
# 4. Let m = (f, g, \epsilon)
# 5. Return m


def domain_specific_generalization(M, G_list, dR, W, num_trials=10):
    """
    Given:
      - M: a MetricMDP
      - G_list: a list of MetricMDPs
      - dR: reward metric (R1, R2) -> float
      - W: distribution metric (μ1, μ2) -> float
    Returns:
      - A MetricMDPMorphism (f†, g†, ε†): M → G† minimizing the error
    """
    best_error = float("inf")

    for G in G_list:
        best_local_error = float("inf")
        best_local_f, best_local_g = None, None

        for _ in range(num_trials):
            # Random maps: f: S_M → S_G, g: A_M → A_G
            f = lambda s: random.choice(list(G.states.keys()))
            g = lambda a: random.choice(
                [act for state in G.states.values() for act in state.actions]
            )

            # Compute E(f, g)
            error = 0
            for state_label in M.states:
                state = M.states[state_label]
                for action_label in state.actions:
                    a1 = action_label
                    a2 = g(a1)

                    try:
                        R1 = state.actions[a1].reward
                        R2 = G.states[f(state_label)].actions[a2].reward

                        T1 = state.actions[a1].measure
                        T2 = G.states[f(state_label)].actions[a2].measure

                        pushed = pushforward(T1, f)
                        e = dR(R1, R2) + W(pushed, T2)
                        error = max(error, e)
                    except KeyError:
                        error = float("inf")
                        break

            if error < best_local_error:
                best_local_error = error
                best_local_f, best_local_g = f, g

        # Track global best
        if best_local_error < best_error:
            G_best = G
            f_best = best_local_f
            g_best = best_local_g
            best_error = best_local_error

    # Compute epsilon function ε†(a)
    def epsilon_fn(a):
        s_label = find_state_of_action(M, a)
        a2 = g_best(a)
        R1 = M.states[s_label].actions[a].reward
        R2 = G_best.states[f_best(s_label)].actions[a2].reward
        T1 = M.states[s_label].actions[a].measure
        T2 = G_best.states[f_best(s_label)].actions[a2].measure
        return dR(R1, R2) + W(pushforward(T1, f_best), T2)

    return MetricMDPMorphism(M, G_best, f_best, g_best, epsilon_fn, dR, W)


def pushforward(measure, f):
    result = {}
    for s, p in measure.items():
        s2 = f(s)
        result[s2] = result.get(s2, 0) + p
    return result


def find_state_of_action(mdp, a_label):
    """
    Find the state label that contains action a_label in mdp.
    Assumes action labels are unique across states.
    """
    for s_label, s in mdp.states.items():
        if a_label in s.actions:
            return s_label
    raise ValueError(f"Action {a_label} not found in any state.")
