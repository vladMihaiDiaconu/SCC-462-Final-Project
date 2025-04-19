import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
# Import the game components
from constants import *
from run import GameController

# Direction conversion Mappings
DIRECTION_TO_INDEX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
INDEX_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_INDEX.items()}
ACTION_DIM = 4

class PacmanWrapper:
    """
    A simple wrapper for the Pac-Man game that handles game control
    and provides a clean interface for reinforcement learning.
    """
    def __init__(self):
        # Initialize game
        self.game = GameController()
        self.game.startGame()

        # Initialize state variables
        self.previous_score = 0
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True

        # Start the game by unpausing
        self.unpause_game()

    def unpause_game(self):
        for _ in range(5):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN}))
            time.sleep(0.1)
            self.game.update()
            if not self.game.pause.paused:
                break

    def check_if_pacman_died(self):
        if self.pacman_was_alive and not self.game.pacman.alive:
            time.sleep(0.5)
            for _ in range(10):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN}))
                time.sleep(0.1)
                self.game.update()
                if self.game.pacman.alive and not self.game.pause.paused:
                    break
        self.pacman_was_alive = self.game.pacman.alive

    def get_state(self):
        """
        Get current game state in a format useful for RL
        Returns a simplified representation of the game state
        """
        self.check_if_pacman_died()
        if self.game.pause.paused:
            self.unpause_game()

        pacman = self.game.pacman
        if pacman.node is None:
            return {
                'position': (0, 0), 
                'direction': STOP, 
                'valid_moves': [],
                'ghosts': [], 
                'score': self.game.score, 
                'lives': self.game.lives
            }

        position = (int(pacman.position.x), int(pacman.position.y))
        # Get valid directions directly for the state_dict
        valid_moves = pacman.validDirections()

        ghosts = [{
            'position': (int(g.position.x), int(g.position.y)),
            'frightened': g.mode.current is FREIGHT
        } for g in self.game.ghosts]

        return {
            'position': position, 
            'direction': pacman.direction, 
            'valid_moves': valid_moves,
            'ghosts': ghosts, 
            'score': self.game.score, 
            'lives': self.game.lives
        }
    
    def take_action(self, direction):
        """
        Execute an action in the game
        action should be one of: UP, DOWN, LEFT, RIGHT (0, 1, 2, 3)
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()

        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()

        pacman = self.game.pacman
        #to check if Pacman is currently moving, if it is moving check the direction it wants to go,
        # if that move is valid, then give it the direction and next node
        if pacman.node and not pacman.target:
            next_node = pacman.node.neighbors.get(direction)
            if next_node:
                pacman.direction = direction
                pacman.target = next_node

        # Update the game
        self.game.update()
        if pacman.target and pacman.position == pacman.target.position:
            pacman.node = pacman.target
            pacman.target = None

        # Get reward
        reward = self.calculate_reward()

        # Get next state
        next_state = self.get_state()

        # Check if episode is done
        done = self.game.lives <= 0 or self.game.pellets.isEmpty()

        return next_state, reward, done


    def calculate_reward(self):
        """Calculate reward based on game events"""
        current_score = self.game.score
        current_pellets = len(self.game.pellets.pelletList)
        current_lives = self.game.lives

        reward = 0

        # Reward for score increase
        reward += 0.1 * (current_score - self.previous_score) 
        # Reward for eating pellets
        reward += 5 * (self.previous_pellets - current_pellets)
        # Penalty for losing lives
        reward -= 50 * (self.previous_lives - current_lives) 
        # Small penalty to encourage faster completion
        reward -= 0.01

        # Reward for completing level
        if self.game.pellets.isEmpty():
            reward += 200

        # Update previous values
        self.previous_score = current_score
        self.previous_pellets = current_pellets
        self.previous_lives = current_lives

        return reward

    def reset(self):
        """Reset the game for a new episode"""
        # Reset game if needed
        if self.game.lives <= 0:
            self.game.restartGame()

        # Make sure game is unpaused
        self.unpause_game()

        #to manually reset.
        pacman = self.game.pacman

        #place pac man at the starting point
        pacman.setPosition()
        #Pac man doesnt start moving until an action is sent to it
        pacman.direction = STOP
        #to remove any in movement in progress
        pacman.target = None

        # Reset state variables
        self.previous_score = self.game.score
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True

        # Return initial state
        return self.get_state()

    def extract_features(self, state):
        """Turn the game states into numbers that neural network can use"""
        #normalize the pac man position
        pos = np.array(state['position']) / 512
        # to convert the directions to a vector
        dir_one_hot = np.zeros(4)
        if state['direction'] in DIRECTION_TO_INDEX:
            dir_one_hot[DIRECTION_TO_INDEX[state['direction']]] = 1
        # extract 3 features per ghost for all the ghosts in the game
        # to tell the neural network where they are and if they are frightened
        ghost_features = []
        for ghost in state['ghosts']:
            ghost_pos = np.array(ghost['position']) / 512
            frightened = np.array([float(ghost['frightened'])])
            ghost_features.extend(np.concatenate((ghost_pos, frightened)))
        # retun a combination of pacmans position, direction and ghosts features
        return np.concatenate((pos, dir_one_hot, ghost_features)).astype(np.float32)

def normalize(values):
    """Normalize a numpy array."""
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / (std + 1e-8)

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01, lam=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.lam = lam # Lambda for GAE

        # Actor network (policy)
        self.actor = self._build_network(output_size=action_size, activation='softmax', name='actor')
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Critic network (value function)
        self.critic = self._build_network(output_size=1, activation=None, name='critic')
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Experience buffer (stores transitions for one update cycle)
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_log_probs = []
        self.buffer_values = []
        self.buffer_next_states = [] # Needed for GAE calculation if done happens mid-buffer
        self.buffer_dones = []      # Needed for GAE calculation

    def _build_network(self, output_size, activation, hidden_units=[128, 128], name=None):
        input_layer = tf.keras.layers.Input(shape=(self.state_size,), name=f'{name}_input')
        x = input_layer
        for i, units in enumerate(hidden_units):
             x = tf.keras.layers.Dense(units, activation='relu', name=f'{name}_dense_{i}')(x)
             x = tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        output_layer = tf.keras.layers.Dense(output_size, activation=activation, name=f'{name}_output')(x)
        return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)


    def choose_action(self, state_features):
        """Chooses an action based on the current state features."""
        state_tensor = tf.convert_to_tensor([state_features], dtype=tf.float32)

        # Predict policy (action probabilities) and value
        policy = self.actor(state_tensor).numpy()[0]
        value = self.critic(state_tensor).numpy()[0, 0]

        #  Masking using valid_moves included in state_features
        valid_moves_start_index = 2 + 4 # pos_dim + dir_dim
        valid_moves_mask = state_features[valid_moves_start_index : valid_moves_start_index + self.action_size]

        # Ensure mask has the correct shape and is boolean/float
        valid_moves_mask = np.array(valid_moves_mask, dtype=np.float32)

        # Apply mask: Set probabilities of invalid actions to near zero
        masked_policy = policy * valid_moves_mask

        # Renormalize the policy if masking occurred and sum is > 0
        policy_sum = np.sum(masked_policy)
        if policy_sum > 1e-8: # Use a small epsilon for floating point comparison
            final_policy = masked_policy / policy_sum
        else:
            # If all valid actions have zero probability (or no valid actions),
            # distribute probability uniformly among the *valid* actions.
            num_valid = np.sum(valid_moves_mask > 0)
            if num_valid > 0:
                final_policy = valid_moves_mask / num_valid
            else:
                # This should ideally not happen if the game always provides valid moves
                print("CRITICAL WARNING: No valid actions available! Defaulting to action 0.")
                final_policy = np.zeros_like(policy)
                final_policy[0] = 1.0 # Default to first action (e.g., UP)

        # Sample action from the final policy
        try:
            action = np.random.choice(self.action_size, p=final_policy)
        except ValueError as e:
             print(f"Error choosing action: {e}")
             print(f"Policy: {policy}")
             print(f"Mask: {valid_moves_mask}")
             print(f"Masked Policy: {masked_policy}")
             print(f"Final Policy: {final_policy}")
             # Fallback: choose a random valid action if possible
             valid_indices = np.where(valid_moves_mask > 0)[0]
             if len(valid_indices) > 0:
                 action = np.random.choice(valid_indices)
                 print(f"Falling back to random valid action: {action}")
             else:
                 action = 0 # Ultimate fallback
                 print(f"Falling back to default action: {action}")

        log_prob = np.log(final_policy[action] + 1e-10)

        return action, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, next_state, done):
        # Store all necessary components for GAE and PPO update
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)
        self.buffer_next_states.append(next_state)
        self.buffer_dones.append(done)

    def learn(self, batch_size=64, epochs=10):
        """Updates the actor and critic networks using data in the buffer."""
        if not self.buffer_states: # Check if buffer is empty
            return # Nothing to learn

        # Prepare data from buffer
        states = np.array(self.buffer_states, dtype=np.float32)
        actions = np.array(self.buffer_actions, dtype=np.int32)
        rewards = np.array(self.buffer_rewards, dtype=np.float32)
        log_probs_old = np.array(self.buffer_log_probs, dtype=np.float32)
        values_old = np.array(self.buffer_values, dtype=np.float32).squeeze()
        dones = np.array(self.buffer_dones, dtype=np.float32)

        # Get the value estimate for the last next_state to bootstrap GAE calculation
        # If the last transition was 'done', the value is 0, otherwise estimate it.
        last_state_tensor = tf.convert_to_tensor([self.buffer_next_states[-1]], dtype=tf.float32)
        last_value = self.critic(last_state_tensor).numpy()[0, 0] if not dones[-1] else 0.0

        # Combine buffer values with the last bootstrapped value
        values_for_gae = np.append(values_old, last_value)
        # Calculate advantages using GAE formula
        advantages = self.compute_gae(rewards, values_for_gae, dones)

        # Calculate returns (target for value function)
        returns = advantages + values_old
        # Normalize advantages
        advantages = normalize(advantages)

        # --- PPO Training Loop ---
        num_samples = len(states)
        indices = np.arange(num_samples)

        for _ in range(epochs): # Iterate multiple times over the same batch
            np.random.shuffle(indices) # Shuffle data each epoch
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get mini-batch
                batch_states = tf.convert_to_tensor(states[batch_indices])
                batch_actions = tf.convert_to_tensor(actions[batch_indices], dtype=tf.int32)
                batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                batch_log_probs_old = tf.convert_to_tensor(log_probs_old[batch_indices], dtype=tf.float32)

                # Perform one training step on the mini-batch
                self._train_step(
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_log_probs_old,
                    batch_returns
                )

        # Clear Buffer after learning
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_log_probs = []
        self.buffer_values = []
        self.buffer_next_states = []
        self.buffer_dones = []

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        batch_size = len(rewards)
        advantages = np.zeros(batch_size, dtype=np.float32)
        last_advantage = 0.0

        # Values array includes the bootstrapped value V(s_T) at index batch_size
        for t in reversed(range(batch_size)):
            mask = 1.0 - dones[t] # 0 if done, 1 if not done
            # Calculate TD error (delta)
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            # Calculate GAE using the recursive formula
            advantages[t] = last_advantage = delta + self.gamma * self.lam * mask * last_advantage

        return advantages

    @tf.function # Decorator compiles this function into a TensorFlow graph for performance
    def _train_step(self, states, actions, advantages, log_probs_old, returns):
        """Performs a single gradient update step for both actor and critic."""
        with tf.GradientTape(persistent=True) as tape: # Persistent tape needed for separate actor/critic grads
            # --- Actor Loss ---
            policy = self.actor(states) # Current policy probabilities
            # Get log probabilities of the actions actually taken, under the *current* policy
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            log_probs_new = tf.gather_nd(tf.math.log(policy + 1e-10), action_indices)

            # Ratio: pi_new(a|s) / pi_old(a|s) == exp(log_prob_new - log_prob_old)
            ratio = tf.exp(log_probs_new - log_probs_old)

            # PPO Clipped Surrogate Objective
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            # Minimum of clipped and unclipped objective, negate for minimization
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # --- Critic Loss ---
            values_new = self.critic(states)[:, 0] # Current value predictions
            # Mean Squared Error between predicted values and calculated returns
            critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - values_new))

            # --- Entropy Loss ---
            # Entropy of the policy distribution
            entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
            entropy_loss = -tf.reduce_mean(entropy) # Negate because we want to maximize entropy

            # --- Total Loss ---
            # Combine losses, applying coefficients
            total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy_loss

        # Calculate gradients
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables) # Actor grads depend on total loss (actor + entropy)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables) # Critic only optimizes critic loss
        del tape # Release tape resources

        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Return losses for monitoring
        # return actor_loss, critic_loss, entropy_loss

    def save(self, actor_path="ppo_actor.keras", critic_path="ppo_critic.keras"):
        """Saves the actor and critic models."""
        try:
            self.actor.save(actor_path)
            self.critic.save(critic_path)
            print(f"PPO agent saved to {actor_path} and {critic_path}")
        except Exception as e:
             print(f"Error saving PPO agent: {e}")

    def load(self, actor_path="ppo_actor.keras", critic_path="ppo_critic.keras"):
        """Loads the actor and critic models."""
        try:
            self.actor = tf.keras.models.load_model(actor_path)
            self.critic = tf.keras.models.load_model(critic_path)
            # Re-assign optimizers if needed
            self.actor_optimizer = self.actor.optimizer if hasattr(self.actor, 'optimizer') else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.critic_optimizer = self.critic.optimizer if hasattr(self.critic, 'optimizer') else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            print(f"PPO agent loaded from {actor_path} and {critic_path}")
        except Exception as e:
            print(f"Error loading PPO agent models from {actor_path} and {critic_path}: {e}. Starting fresh.")


def train_ppo(agent, env, episodes=1000, max_steps=2000, update_frequency=2048, epochs_per_update=10, save_interval=50):
    """Train the PPO agent on the Pac-Man game."""
    # Training metrics
    episode_rewards = []
    moving_avg = []
    best_avg_reward = -np.inf
    global_step = 0 # Total steps across all episodes

    print(f"Starting PPO Training: Episodes={episodes}, MaxSteps/Ep={max_steps}, UpdateFreq={update_frequency}, Epochs/Update={epochs_per_update}")

    for episode in range(episodes):
        # Reset environment and get initial state dictionary
        state_dict = env.reset()
        # Extract features from the state dictionary
        state_features = env.extract_features(state_dict)

        # Initialize episode variables
        episode_reward = 0
        episode_steps = 0
        done = False

        # Step loop
        while not done and episode_steps < max_steps:
            # Get action, log_prob, value from PPO agent using current features
            action_index, log_prob, value = agent.choose_action(state_features)
            # Convert action index to game direction
            action_direction = INDEX_TO_DIRECTION[action_index]

            # Environment step: take action, get next state_dict, reward, done
            next_state_dict, reward, done = env.take_action(action_direction)
            # Extract features for the next state
            next_state_features = env.extract_features(next_state_dict)

            # Store transition in PPO buffer
            agent.store_transition(
                state=state_features,
                action=action_index,
                reward=reward,
                log_prob=log_prob,
                value=value,
                next_state=next_state_features,
                done=done
            )

            # Update state variables for the next iteration
            state_features = next_state_features
            state_dict = next_state_dict

            # Accumulate rewards and increment counters
            episode_reward += reward
            episode_steps += 1
            global_step += 1

            # Learn if the buffer reaches the update frequency
            if len(agent.buffer_states) >= update_frequency:
                agent.learn(batch_size=update_frequency // 4, epochs=epochs_per_update)

        # --- End of Step Loop ---

        # --- End of Episode ---
        episode_rewards.append(episode_reward)
        # Calculate moving average
        avg_window = 100
        current_avg = np.mean(episode_rewards[-avg_window:]) if len(episode_rewards) >= avg_window else np.mean(episode_rewards)
        moving_avg.append(current_avg)

        # Save best model based on moving average (only after enough episodes for meaningful avg)
        if len(episode_rewards) > avg_window // 2 and current_avg > best_avg_reward:
            best_avg_reward = current_avg
            print(f"*** New Best Average Reward ({avg_window} ep): {best_avg_reward:.2f} at episode {episode+1} ***")
            agent.save("best_actor.keras", "best_critic.keras")

        # Progress reporting
        if (episode + 1) % 10 == 0 or episode == 0: # Report every 10 episodes
             print(f"Ep {episode + 1} | "
                   f"Score: {state_dict['score']} | " # Use final score from state_dict
                   f"Reward: {episode_reward:.1f} | "
                   f"Avg {avg_window}: {current_avg:.1f} | "
                   f"Steps: {episode_steps} | "
                   f"Total Steps: {global_step}") # Show global steps

        # Checkpoint saving
        if (episode + 1) % save_interval == 0:
             print(f"--- Saving checkpoint at episode {episode + 1} ---")
             agent.save(f"checkpoint_{episode + 1}_actor.keras",
                        f"checkpoint_{episode + 1}_critic.keras")

    # --- End of Training Loop ---

    # Final save and plot
    print("--- Training finished. Saving final model and plot. ---")
    agent.save("final_actor.keras", "final_critic.keras")

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.4, label='Episode Reward', color='lightblue')
    plt.plot(moving_avg, linewidth=2, label=f'{avg_window}-Episode Avg Reward', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Progress on Pac-Man")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("ppo_training_progress.png")
    plt.close()
    print("Training progress plot saved to ppo_training_progress.png")

    return agent


if __name__ == "__main__":
    # Configuration
    episodes = 1000                # Total number of episodes to train
    max_steps_per_episode = 3000    # Max steps allowed in one episode
    ppo_update_frequency = 2048     # Number of steps to collect before updating networks
    ppo_epochs_per_update = 10      # Number of optimization epochs over the collected data
    save_checkpoint_interval = 100  # Save model weights every X episodes

    # --- Environment Setup ---
    print("Initializing Pygame and Pacman Environment...")
    pygame.init() # Ensure Pygame is initialized
    env = PacmanWrapper()
    print("Environment Initialized.")

    # --- Agent Setup ---
    # Get state and action dimensions dynamically from the environment
    _initial_state_dict = env.reset() # Get initial raw state
    sample_state_features = env.extract_features(_initial_state_dict) # Get feature vector
    state_size = sample_state_features.shape[0]
    action_size = ACTION_DIM

    print(f"State features dimension: {state_size}")
    print(f"Action dimension: {action_size}")

    # Create the PPO agent
    ppo_agent = PPOAgent(state_size, action_size, learning_rate=0.0001, clip_epsilon=0.1, entropy_coeff=0.01)

    # Load pre-trained model if available
    try:
        ppo_agent.load("final_actor.keras", "final_critic.keras")
        # Or load the best one:
        # ppo_agent.load("best_actor.keras", "best_critic.keras")
    except Exception as e:
         print(f"Could not load existing models, starting training from scratch. Error: {e}")


    # --- Start Training ---
    print(f"\n--- Starting PPO Training ---")
    trained_ppo_agent = train_ppo(
        agent=ppo_agent,
        env=env,
        episodes=episodes,
        max_steps=max_steps_per_episode,
        update_frequency=ppo_update_frequency,
        epochs_per_update=ppo_epochs_per_update,
        save_interval=save_checkpoint_interval
    )

    print("\n--- PPO Training Complete ---")

    # Close Pygame window if it's still open
    pygame.quit()