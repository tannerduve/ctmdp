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
                else:  # deterministic action
                    measure = {action_data: 1}
                    action = Action(state, measure)

                state.actions[action_label] = action
        self.start = self.first_state
        self.goals = [self.last_state]

    @property
    def state_labels(self):
        return {v: k for k, v in self.states.items()}

    def relabel_states(self, relabeling_tuples):
        for old, new in relabeling_tuples:
            if old in self.states:
                state = self.states[old]
                self.states.pop(old)
                self.states[new] = state

    def relabel_actions(self, relabeling_tuples):
        for old, new in relabeling_tuples:
            for state in self.states.values():
                if old in state.actions:
                    action = state.actions[old]
                    state.actions.pop(old)
                    state.actions[new] = action

    def relabel_all_states(self, relabeling_function):
        old_states = self.states.copy()
        for old in old_states:
            state = self.states[old]
            new = relabeling_function(old)
            self.states.pop(old)
            self.states[new] = state

    def simulate_policy(self, policy, start=None, goals=None, max_steps=50):
        if start is None:
            start = self.start
        if goals is None:
            goals = self.goals
        states = [start]
        actions = []
        while len(states) - 1 <= max_steps and not states[-1].label in goals:
            actions.append(policy.select_action(states[-1].label))
            states.append(actions[-1].transition())
        return Path(self, states, actions)

    def set_rewards(self, f):
        for state in self.states.values():
            for action in state.actions.values():
                action.reward = f(action)

    def add_rewards(self, f):
        for state in self.states.values():
            for action in state.actions.values():
                action.reward += f(action)

    @property
    def first_state(self):
        return list(self.states.values())[0]

    @property
    def last_state(self):
        return list(self.states.values())[-1]

    def set_goal_states(self, goals):
        self.goals = goals

    def set_start_state(self, start):
        self.start = start


class Path:
    def __init__(self, mdp, states, actions):
        self.mdp = mdp
        self.states = states
        self.actions = actions

    @property
    def transitions(self):
        return [
            (self.states[i], self.actions[i], self.states[i + 1])
            for i in range(len(self.actions))
        ]

    def __repr__(self):
        return "".join(
            [f"{s.label} -{a.label}-> " for s, a, _ in self.transitions]
        ) + str(self.transitions[-1][-1].label)


class Policy:
    def __init__(self, mdp, policy_dict):
        self.mdp = mdp
        # Drop policy for goal states.
        self.policy = {
            state: actions
            for state, actions in policy_dict.items()
            if not state in map(State.label.fget, mdp.goals)
        }

    def select_action(self, state_label):
        action_probs = self.policy[state_label]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = choices(actions, weights=probs)[0]
        return self.mdp.states[state_label].actions[action]

    @property
    def deterministic(self):
        """Return a deterministic version of the policy
        by taking the actions with the heighest weight."""
        policy = {
            state: dict([max(actions.items(), key=lambda t: t[1])])
            for state, actions in self.policy.items()
        }
        return Policy(self.mdp, policy)


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
