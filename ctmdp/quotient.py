from collections import defaultdict
import ctmdp


def bisimulation_partition_refinement(mdp):
    """
    Build a bisimulation-based partition of the given MDP.
    Returns a list of blocks, where each block is a set of state labels.
    """

    # 1) INITIAL PARTITION: group states that have the same set of actions,
    initial_groups = defaultdict(set)

    for s_label, s_obj in mdp.states.items():
        # Build a basic signature from the action set
        action_set = tuple(sorted(s_obj.actions.keys()))
        initial_groups[action_set].add(s_label)

    partition = list(initial_groups.values())

    # 2) ITERATE refinement until stable
    changed = True
    while changed:
        changed = False
        new_partition = []

        for block in partition:
            # block is a set of states
            if len(block) == 1:
                # singletons can't split further
                new_partition.append(block)
                continue

            # Group states in 'block' by their transition signatures
            # i.e. which block(s) they transition to, with what probabilities, for each action
            signatures = defaultdict(set)

            # We need a stable ordering of action labels to produce a consistent signature
            # for each state.
            # We'll just sort by action_label for reproducibility.
            # If MDP states can have different sets of actions, we already separated them
            # in the initial partition, but let's be safe here too.
            representative_state = mdp.states[next(iter(block))]
            all_actions_in_block = sorted(representative_state.actions.keys())

            for s_label in block:
                s_obj = mdp.states[s_label]

                # If for some reason states in the same block differ in actions,
                # the signature will differ anyway, so they will be split.
                action_labels = sorted(s_obj.actions.keys())

                # Build the signature for s_label
                sig_for_s = []
                for a_label in action_labels:
                    a_obj = s_obj.actions[a_label]

                    # Probability distribution over partition blocks
                    # We'll sum up the probability that a_obj goes to each block in 'partition'
                    prob_vector = []
                    for blk in partition:
                        blk_prob = sum(
                            prob
                            for (t_state, prob) in a_obj.measure.items()
                            if t_state in blk
                        )
                        prob_vector.append(blk_prob)

                    # Reward
                    rew = a_obj.reward

                    sig_for_s.append((a_label, tuple(prob_vector)))  # omit reward

                # Sort so the order of actions doesn't matter
                sig_for_s.sort(key=lambda x: x[0])
                sig_for_s = tuple(sig_for_s)

                signatures[sig_for_s].add(s_label)

            # Now 'signatures' might have split the block into multiple subsets
            if len(signatures) == 1:
                # No split
                new_partition.append(block)
            else:
                # We have a real split
                changed = True
                new_partition.extend(signatures.values())

        partition = new_partition

    return partition


def build_quotient_mdp(mdp):
    """
    Build a quotient MDP under the bisimulation partition.
    Returns (quotient_mdp, state_to_block, g) where:
      - quotient_mdp is an MDP object
      - state_to_block is a dict mapping each original state to its block ID
      - g(action_label) is the identity mapping on actions (for demonstration)
    """
    partition = bisimulation_partition_refinement(mdp)

    # Step 1: Map each state to its block ID (canonical surjection f)
    state_to_block = {}
    blocks = []
    for block_id, block in enumerate(partition):
        blocks.append(block)
        for s in block:
            state_to_block[s] = block_id

    # Step 2: Define identity action morphism g
    def g(action_label):
        return action_label

    # Step 3: Build quotient MDP description
    quotient_desc = {}
    for block_id, block in enumerate(blocks):
        # pick a representative
        rep_state_label = next(iter(block))
        rep_state = mdp.states[rep_state_label]

        action_dict = {}
        for a_label, a_obj in rep_state.actions.items():
            # Probability distribution over *block* IDs
            transition_to_blocks = defaultdict(float)
            for t_state, prob in a_obj.measure.items():
                t_block = state_to_block[t_state]
                transition_to_blocks[t_block] += prob

            # Convert to standard dict
            transition_dict = dict(transition_to_blocks)
            transition_dict["reward"] = a_obj.reward
            action_dict[a_label] = transition_dict

        quotient_desc[block_id] = action_dict

    # Finally, construct the actual MDP
    quotient_mdp = ctmdp.mdp.MDP(quotient_desc)
    return quotient_mdp, state_to_block, g
