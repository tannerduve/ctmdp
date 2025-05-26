from random import choices


class State:
    def __init__(self, mdp):
        self.mdp = mdp
        self.actions = {}

    @property
    def label(self):
        return self.mdp.state_labels[self]

    @property
    def action_labels(self):
        return {v: k for k, v in self.actions.items()}

    def __repr__(self):
        return f"<State {self.label}>"


class Action:
    def __init__(self, state, measure, reward=0):
        self.state = state
        self.measure = measure
        self.reward = reward
        self.mdp = state.mdp

    @property
    def label(self):
        return self.state.action_labels[self]

    def transition(self):
        states = list(self.measure.keys())
        weights = list(self.measure.values())
        outcome = choices(states, weights=weights)[0]
        return self.mdp.states[outcome]

    @property
    def reachable_states(self):
        return [
            self.mdp.states[state]
            for state, weight in self.measure.items()
            if weight > 0
        ]

    def __repr__(self):
        return f"<Action {self.label} @{self.state}>"


class MDP:
    def __init__(self, description):
        self.states = {}
        for state_label, state_actions_desc in description.items():
            state = State(self)
            self.states[state_label] = state
            for action_label, action_data in state_actions_desc.items():
                if isinstance(action_data, tuple):  # description with reward
                    measure, reward = action_data
                    action = Action(state, measure, reward=reward)
                elif isinstance(action_data, dict):  # description without reward
                    measure = action_data
                    action = Action(state, measure)

                state.actions[action_label] = action

    @property
    def state_labels(self):
        return {v: k for k, v in self.states.items()}


class Policy:
    def __init__(self, mdp, policy_dict):
        self.mdp = mdp
        self.policy = policy_dict

    def select_action(self, state_label):
        action_probs = self.policy[state_label]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = choices(actions, weights=probs)[0]
        return self.mdp.states[state_label].actions[action]


class MDPMorphism:
    def __init__(self, source, target, f, g):
        self.source = source  # MDP instance
        self.target = target  # MDP instance
        self.f = f  # function: state_label1 -> state_label2
        self.g = g  # function: action_label1 -> action_label2

    def pushforward(self, measure):
        """Pushforward of a distribution under f: S1 → S2"""
        result = {}
        for s1, prob in measure.items():
            s2 = self.f(s1)
            result[s2] = result.get(s2, 0) + prob
        return result

    def is_valid(self):
        for s_label1, s_obj1 in self.source.states.items():
            for a_label1, a_obj1 in s_obj1.actions.items():
                a_label2 = self.g(a_label1)
                s_label2 = self.f(s_label1)

                try:
                    a_obj2 = self.target.states[s_label2].actions[a_label2]
                except KeyError:
                    return False  # mapped action not defined at mapped state

                # Check reward compatibility
                if a_obj1.reward != a_obj2.reward:
                    return False

                # Check transition compatibility
                pushed = self.pushforward(a_obj1.measure)
                if not self._measures_equal(pushed, a_obj2.measure):
                    return False
        return True

    def _measures_equal(self, μ1, μ2, tol=1e-6):
        keys = set(μ1.keys()).union(μ2.keys())
        return all(abs(μ1.get(k, 0) - μ2.get(k, 0)) < tol for k in keys)
