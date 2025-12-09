//! # Q-Learning
//!
//! The `q_learning` module contains implementations of temporal difference learning algorithms
//! (SARSA and Q-Learning) for MDPs.

use madepro::models::{Sampler, Config, ActionValue};
use crate::mdp::MDP;
use crate::error::Error;

/// Internal helper function that implements both SARSA and Q-Learning
/// The `q_learning` parameter determines which algorithm to use:
/// - `true` for Q-Learning (off-policy)
/// - `false` for SARSA (on-policy)
fn sarsa_q_learning<M>(
    mdp: &M,
    config: &Config,
    q_learning: bool,
) -> Result<ActionValue<M::State, M::Action>, Error>
where
    M: MDP,
    M::State: Clone,
    M::Action: Clone,
{
    let states = mdp.all_states();
    let state_action_pairs = mdp.all_state_action_pairs();
    
    // Collect all unique actions across all states
    // Use a HashSet to deduplicate without requiring Ord
    use std::collections::HashSet;
    let all_actions: Vec<M::Action> = state_action_pairs
        .iter()
        .map(|(_, a)| a.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let actions: Sampler<M::Action> = all_actions.into();
    
    let mut action_value = ActionValue::new(states, &actions);
    
    for _ in 0..config.num_episodes {
        // Start from a random state
        let mut state = states.get_random().clone();
        
        // Get available actions at this state
        let available_actions = mdp.actions_at(&state);
        if available_actions.is_empty() {
            continue;
        }
        
        // Select initial action using epsilon-greedy
        let mut action = action_value
            .epsilon_greedy(&actions, &state, config.exploration_rate)
            .clone();
        
        for _ in 0..config.max_num_steps {
            // Transition: get measure over next states and reward
            let (measure, reward) = mdp.stochastic_transition(&state, &action)?;
            
            // Sample next state from the measure
            let next_state = match measure.sample() {
                Some(s) => s.clone(),
                None => {
                    // If measure is empty, stay in current state
                    state.clone()
                }
            };
            
            // Get available actions at next state
            let next_available_actions = mdp.actions_at(&next_state);
            if next_available_actions.is_empty() {
                break;
            }
            
            // Select next action (for SARSA) or greedy action (for Q-Learning)
            let next_action = if q_learning {
                // Q-Learning: use greedy action for target
                action_value.greedy(&next_state).clone()
            } else {
                // SARSA: use epsilon-greedy action
                action_value
                    .epsilon_greedy(&actions, &next_state, config.exploration_rate)
                    .clone()
            };
            
            // Update Q-value using Bellman equation
            let current_q = action_value.get(&state, &action);
            let next_q = action_value.get(&next_state, &next_action);
            let target = reward + config.discount_factor * next_q;
            let new_q = current_q + config.learning_rate * (target - current_q);
            
            action_value.insert(&state, &action, new_q);
            
            // Move to next state
            state = next_state;
            action = next_action;
            
            // Check if we've reached a terminal state
            if mdp.is_final_state(&state) {
                break;
            }
        }
    }
    
    Ok(action_value)
}

/// # SARSA
///
/// This function implements the SARSA (State-Action-Reward-State-Action) algorithm.
/// It is an on-policy temporal difference learning algorithm.
///
/// The algorithm works by:
/// 1. Starting from a random state
/// 2. Selecting actions using an epsilon-greedy policy
/// 3. Updating Q-values based on the actual action taken (on-policy)
/// 4. Stopping after the given number of episodes or when reaching a terminal state
///
/// # Arguments
/// * `mdp` - The MDP to learn from
/// * `config` - Configuration parameters (learning rate, discount factor, exploration rate, etc.)
///
/// # Returns
/// An `ActionValue` table containing the learned Q-values for all state-action pairs
pub fn sarsa<M>(mdp: &M, config: &Config) -> Result<ActionValue<M::State, M::Action>, Error>
where
    M: MDP,
    M::State: Clone,
    M::Action: Clone,
{
    sarsa_q_learning(mdp, config, false)
}

/// # Q-Learning
///
/// This function implements the Q-Learning algorithm.
/// It is an off-policy temporal difference learning algorithm.
///
/// The algorithm works by:
/// 1. Starting from a random state
/// 2. Selecting actions using an epsilon-greedy policy
/// 3. Updating Q-values based on the greedy action (off-policy)
/// 4. Stopping after the given number of episodes or when reaching a terminal state
///
/// Unlike SARSA, Q-Learning uses the greedy action for the target value,
/// making it off-policy and often more efficient.
///
/// # Arguments
/// * `mdp` - The MDP to learn from
/// * `config` - Configuration parameters (learning rate, discount factor, exploration rate, etc.)
///
/// # Returns
/// An `ActionValue` table containing the learned Q-values for all state-action pairs
pub fn q_learning<M>(mdp: &M, config: &Config) -> Result<ActionValue<M::State, M::Action>, Error>
where
    M: MDP,
    M::State: Clone,
    M::Action: Clone,
{
    sarsa_q_learning(mdp, config, true)
}
