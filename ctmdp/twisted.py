import ctmdp
from collections import deque


class Automaton:
    def __init__(
        self, states, alphabet, transitions, initial, accepting=None, sink=None
    ):
        """
        states: list of automaton states
        alphabet: list (or set) of symbols (labels)
        transitions: dict mapping (state, symbol) -> next_state
        initial: initial state
        accepting: set of accepting states (optional, for reference)
        sink: optional sink state for undefined transitions (e.g., error state)
        """
        self.states = states
        self.alphabet = set(alphabet)
        self.transitions = transitions
        self.initial = initial
        self.accepting = set(accepting) if accepting else set()
        self.sink = sink

    def delta(self, q, sigma):
        """Deterministic transition: if (q, sigma) in transitions, return that;
        else if sink defined and q != sink, go to sink; otherwise None."""
        if (q, sigma) in self.transitions:
            return self.transitions[(q, sigma)]
        elif self.sink is not None:
            return self.sink if q != self.sink else self.sink
        else:
            return None


class TwistedMDP:
    def __init__(self, base_mdp, automaton, label_func):
        self.base_mdp = base_mdp
        self.automaton = automaton
        self.states = {}  # Twisted states labeled by (base_state_label, automaton_state_label)
        self._build_twisted_mdp(label_func)

    def _build_twisted_mdp(self, label_func):
        initial_pair = (
            self.base_mdp.states[next(iter(self.base_mdp.states))].label,
            self.automaton.initial,
        )
        queue = deque([initial_pair])
        visited = set()

        while queue:
            (base_label, auto_label) = queue.popleft()
            if (base_label, auto_label) in visited:
                continue
            visited.add((base_label, auto_label))

            base_state = self.base_mdp.states[base_label]
            twisted_state = ctmdp.mdp.State((base_label, auto_label))
            twisted_state.actions = {}

            for action_label, action in base_state.actions.items():
                measure = {}
                for target_label, prob in action.measure.items():
                    sigma = label_func(target_label)
                    auto_next = self.automaton.delta(auto_label, sigma)
                    if auto_next is not None:
                        twisted_target_label = (target_label, auto_next)
                        measure[twisted_target_label] = prob
                        if twisted_target_label not in visited:
                            queue.append(twisted_target_label)
                if measure:
                    twisted_action = ctmdp.mdp.Action(
                        action_label, twisted_state, measure, action.reward
                    )
                    twisted_action.mdp = self
                    twisted_state.actions[action_label] = twisted_action

            self.states[twisted_state.label] = twisted_state

        self.initial = self.states[initial_pair]

    def counit(self, twisted_state_label):
        base_label, _ = twisted_state_label
        return self.base_mdp.states[base_label]

    def coextend_morphism(self, state_map, action_map):
        def extended_state_map(twisted_state_label):
            base_label, auto_label = twisted_state_label
            mapped_base_label = state_map(base_label)
            return (mapped_base_label, auto_label)

        def extended_action_map(action_label):
            return action_map(action_label)

        return extended_state_map, extended_action_map


# Automaton as provided:
class Automaton:
    def __init__(
        self, states, alphabet, transitions, initial, accepting=None, sink=None
    ):
        self.states = states
        self.alphabet = set(alphabet)
        self.transitions = transitions
        self.initial = initial
        self.accepting = set(accepting) if accepting else set()
        self.sink = sink

    def delta(self, q, sigma):
        if (q, sigma) in self.transitions:
            return self.transitions[(q, sigma)]
        elif self.sink is not None:
            return self.sink if q != self.sink else self.sink
        else:
            return None
