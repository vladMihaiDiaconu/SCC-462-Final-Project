import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from constants import *
from run import GameController

# Direction conversion mappings
DIRECTION_TO_INDEX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
INDEX_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_INDEX.items()}
ACTION_DIM = 4

class PacmanWrapper:
    """
    A wrapper around the Pac-Man game to provide a clean interface suitable for reinforcement learning agents.
    It handles game state extraction, action execution, reward calculation, 
    and game lifecycle management.
    """
    def __init__(self):
        # Initializing game controller
        self.game = GameController()
        # Starting the game logic
        self.game.startGame()

        # Initializing state variables for reward calculation
        # Storing previous state values to calculate changes
        self.previous_score = 0
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_powerpellets = len(self.game.pellets.powerpellets)
        self.previous_lives = self.game.lives
        # Tracking if Pacman was alive in the previous step
        self.pacman_was_alive = True

        # Start the game by unpausing
        self.unpause_game()

    def unpause_game(self):
        """
        Unpause the game by simulating key presses (SPACE or RETURN).
        """
        for _ in range(5):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN}))
            time.sleep(0.1)
            self.game.update()
            if not self.game.pause.paused:
                break

    def check_if_pacman_died(self):
        """
        Checks if Pacman died since the last check. 
        If so, attempts to simulate key presses to continue the game after death.
        """
        # Detect if Pac-Man just died
        if self.pacman_was_alive and not self.game.pacman.alive:
            # Wait a bit for death animation
            time.sleep(0.5)
            # Try to unpause repeatedly after death
            for _ in range(10):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN}))
                time.sleep(0.1)
                self.game.update()
                # If game unpaused, stop
                if self.game.pacman.alive and not self.game.pause.paused:
                    break
        # Update alive status for next check
        self.pacman_was_alive = self.game.pacman.alive

    def get_state(self):
        """
        Gets the current game state as a dictionary containing relevant information.
        Handles checks for death and pauses before extracting the state.
        Returns:
            dict: A dictionary representing the simplified game state.
                  Returns a default state if Pacman's node is invalid.
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()

        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()

        pacman = self.game.pacman
        # Handling potential case where Pacman might not be on a valid node
        if pacman.node is None:
            return {
                'position': (0, 0), 
                'direction': STOP, 
                'valid_moves': [],
                'ghosts': [], 
                'score': self.game.score, 
                'lives': self.game.lives,
                'level': self.game.level
            }
        # Pacman's current position
        position = (int(pacman.position.x), int(pacman.position.y))
        # Get valid directions Pacman can move into from the current node
        valid_moves = pacman.validDirections()

        # Information about each ghost
        ghosts = [{
            'position': (int(g.position.x), int(g.position.y)),
            'frightened': g.mode.current is FREIGHT
        } for g in self.game.ghosts]

        # Return the state dictionary
        return {
            'position': position, 
            'direction': pacman.direction, 
            'valid_moves': valid_moves,
            'ghosts': ghosts, 
            'score': self.game.score, 
            'lives': self.game.lives,
            'level': self.game.level
        }
    
    def take_action(self, direction):
        """
        Execute a given direction action, update the game state,
        calculate the reward, and return the results.
        Args:
            direction: The direction constant (UP, DOWN, LEFT, RIGHT) to execute.

        Returns:
            tuple: (next_state_dict, reward, done) containing the next game state,
                   the calculated reward, and a boolean indicating if the episode ended.
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()

        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()

        pacman = self.game.pacman
        if pacman.node and not pacman.target:
            # Check if the chosen direction leads to a valid neighboring node
            # Set new direction and target node
            next_node = pacman.node.neighbors.get(direction)
            if next_node:
                pacman.direction = direction
                pacman.target = next_node

        # Update the game
        self.game.update()
        if pacman.target and pacman.position == pacman.target.position:
            pacman.node = pacman.target
            pacman.target = None

        # Calculate the reward
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
        current_powerpellets = len(self.game.pellets.powerpellets)
        current_lives = self.game.lives

        reward = 0

        # Significant penalty for losing a life
        reward -= 100 * (self.previous_lives - current_lives)

        # Small penalty to encourage faster completion
        reward -= 0.01

        # Reward for eating pellets
        reward += 5 * (self.previous_pellets - current_pellets)

        # Reward for score increase
        reward += 0.1 * (current_score - self.previous_score)

        # Reward for eating a power pellet
        if current_powerpellets < self.previous_powerpellets:
            reward += 50

        # Reward for eating a frightened ghost
        if self.game.ghost_eaten_in_last_step:
            reward += 75

        # Penalty for being in close proximity to an aggressive ghost
        # This encourages the agent to maintain safe distances.
        proximity_penalty = 0
        for ghost in self.game.ghosts:
            if ghost.mode.current == CHASE or ghost.mode.current == SCATTER:
                if (self.game.pacman.position - ghost.position).magnitude() < SAFE_DISTANCE:
                    proximity_penalty += 1 # Accumulate penalty for each close aggressive ghost
        reward -= proximity_penalty * 1.0

        # Reward for completing level
        if self.game.pellets.isEmpty():
            reward += 200

        # Update previous values
        self.previous_score = current_score
        self.previous_pellets = current_pellets
        self.previous_lives = current_lives
        self.previous_powerpellets = current_powerpellets

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
        self.previous_powerpellets = len(self.game.pellets.powerpellets)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True

        # Return initial state
        return self.get_state()
    
    def extract_features(self, state_dict):
        """Converts state dict into a normalized numerical vector."""
        # Max coordinate values for normalization
        max_x, max_y = 560, 620
        # Normalized position [x, y] in range [0, 1]
        pos = np.array(state_dict['position']) / np.array([max_x, max_y])
        # One-hot encoded direction [UP, DOWN, LEFT, RIGHT]
        dir_one_hot = np.zeros(ACTION_DIM)
        if state_dict['direction'] in DIRECTION_TO_INDEX:
            dir_one_hot[DIRECTION_TO_INDEX[state_dict['direction']]] = 1
        # One-hot encoded valid moves (binary mask) [UP, DOWN, LEFT, RIGHT]
        valid_moves_one_hot = np.zeros(ACTION_DIM)
        for move in state_dict['valid_moves']:
            if move in DIRECTION_TO_INDEX:
                valid_moves_one_hot[DIRECTION_TO_INDEX[move]] = 1
        
        # Extract features for each ghost, padding if fewer than 4 exist
        ghost_features = []
        num_ghosts = 4
        for i in range(num_ghosts):
            if i < len(state_dict['ghosts']):
                ghost = state_dict['ghosts'][i]
                # Normalized position [x, y]
                ghost_pos = np.array(ghost['position']) / np.array([max_x, max_y])
                # Frightened status [0.0 or 1.0]
                frightened = np.array([float(ghost['frightened'])])
                # Combine pos + frightened for this ghost
                ghost_features.extend(np.concatenate((ghost_pos, frightened)))
            else:
                # Pad with placeholder values (-1) if ghost doesn't exist
                ghost_features.extend([-1.0, -1.0, -1.0])
        # Concatenate all feature components into a single numpy array
        features = np.concatenate((
            pos,                   # 2 features
            dir_one_hot,           # 4 features
            valid_moves_one_hot,   # 4 features
            ghost_features         # 12 features (4 ghosts * 3 features/ghost)
        )).astype(np.float32)
        
        return features

def predict_ghost_positions(self, steps_ahead=3):
    """
    Predict ghost movement for the next few steps to enable proactive avoidance.
    This is a simple prediction based on current direction and maze constraints.
    
    Args:
        steps_ahead: How many steps to predict ahead
        
    Returns:
        List of predicted ghost positions for each step
    """
    predicted_positions = []
    
    for step in range(1, steps_ahead + 1):
        step_predictions = []
        
        for ghost in self.game.ghosts:
            # Skip frightened ghosts as they're not threatening
            if ghost.mode.current is FREIGHT:
                step_predictions.append(None)
                continue
                
            # If ghost is at a node, predict possible directions
            if ghost.node is not None:
                # For simplicity, predict ghost will continue in current direction if possible
                # This is a simplification - actual ghost AI is more complex
                current_pos = (ghost.position.x, ghost.position.y)
                
                # Get neighboring nodes in 4 directions
                possible_next_positions = []
                for direction in [UP, DOWN, LEFT, RIGHT]:
                    next_node = ghost.node.neighbors.get(direction)
                    if next_node:
                        # Ghosts don't turn around unless they have to
                        # Check if this would be a reversal of direction
                        reversal = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}.get(ghost.direction)
                        if direction != reversal or len(ghost.node.neighbors) == 1:
                            vec = next_node.position - ghost.position
                            # Calculate how far ghost would travel in 'step' steps at its speed
                            # This is an approximation assuming constant speed
                            step_distance = ghost.speed * step
                            # Scale the direction vector by the distance
                            if vec.magnitude() > 0:  # Avoid division by zero
                                scaled_vec = vec * (step_distance / vec.magnitude())
                                new_pos = (ghost.position.x + scaled_vec.x, 
                                          ghost.position.y + scaled_vec.y)
                                possible_next_positions.append(new_pos)
                
                # If multiple positions are possible, use the one closest to Pac-Man for worst-case scenario
                if possible_next_positions:
                    pacman_pos = (self.game.pacman.position.x, self.game.pacman.position.y)
                    distances = [((pos[0] - pacman_pos[0])**2 + (pos[1] - pacman_pos[1])**2) for pos in possible_next_positions]
                    closest_idx = distances.index(min(distances))
                    step_predictions.append(possible_next_positions[closest_idx])
                else:
                    # If no valid moves, ghost stays in place
                    step_predictions.append(current_pos)
            else:
                # Ghost is between nodes, extrapolate position based on velocity
                new_x = ghost.position.x + ghost.direction.x * ghost.speed * step
                new_y = ghost.position.y + ghost.direction.y * ghost.speed * step
                step_predictions.append((new_x, new_y))
                
        predicted_positions.append(step_predictions)
    
    return predicted_positions

def calculate_safety_map(self, state_dict):
    """
    Create a danger map that considers path safety rather than just proximity.
    Evaluates the surrounding area's safety level for escape options.
    
    Returns:
        safety_score: A safety score where higher values mean safer positions
    """
    # Get Pac-Man's current node
    pacman = self.game.pacman
    if pacman.node is None:
        return 0.0  # Not on a node, can't evaluate safety
        
    # Count escape routes from current position
    escape_routes = 0
    for direction in [UP, DOWN, LEFT, RIGHT]:
        if pacman.node.neighbors.get(direction):
            escape_routes += 1
            
    # Evaluate if Pac-Man is in a dead end
    is_dead_end = escape_routes <= 1
    
    # Calculate distance to nearest power pellet (if any)
    power_pellet_distances = []
    for pellet in self.game.pellets.powerpellets:
        pellet_pos = (pellet.position.x, pellet.position.y)
        pacman_pos = (pacman.position.x, pacman.position.y)
        distance = ((pellet_pos[0] - pacman_pos[0])**2 + (pellet_pos[1] - pacman_pos[1])**2)**0.5
        power_pellet_distances.append(distance)
    
    nearest_power_pellet = min(power_pellet_distances) if power_pellet_distances else float('inf')
    
    # Get predicted ghost positions
    predicted_ghost_positions = self.predict_ghost_positions(steps_ahead=3)
    
    # Calculate safety based on predicted ghost positions
    safety_score = 100.0  # Start with a high safety score
    
    # Reduce safety score for each ghost based on predicted positions
    for step, positions in enumerate(predicted_ghost_positions):
        step_penalty = 1.0 / (step + 1)  # Nearer steps are more dangerous
        
        for i, pos in enumerate(positions):
            if pos is None:  # Skip frightened ghosts
                continue
                
            # Get ghost info
            ghost = self.game.ghosts[i]
            
            # Calculate distance to predicted ghost position
            pacman_pos = (pacman.position.x, pacman.position.y)
            ghost_distance = ((pos[0] - pacman_pos[0])**2 + (pos[1] - pacman_pos[1])**2)**0.5
            
            # Penalty based on distance and ghost type
            ghost_weight = self.get_ghost_danger_weight(ghost)
            distance_penalty = 50.0 / (ghost_distance + 10.0)  # Higher penalty for closer ghosts
            
            safety_score -= distance_penalty * ghost_weight * step_penalty
    
    # Dead ends are very dangerous
    if is_dead_end:
        safety_score *= 0.5
    
    # Having multiple escape routes is safer
    safety_score *= (1.0 + 0.2 * escape_routes)
    
    # Being near a power pellet increases safety
    if nearest_power_pellet < 100:
        safety_score += 50.0 / (nearest_power_pellet + 1.0)
        
    return max(0.0, safety_score)  # Ensure non-negative

def get_ghost_danger_weight(self, ghost):
    """
    Calculate danger weight for specific ghost types based on their behavior patterns.
    
    In the original Pac-Man:
    - Blinky (red): Most aggressive, follows Pac-Man directly
    - Pinky (pink): Tries to ambush by targeting ahead of Pac-Man
    - Inky (cyan): Complex targeting using Blinky's position
    - Clyde (orange): Moves randomly when far, follows when close
    
    Returns:
        float: Danger weight for the ghost
    """
    # Check if this is a specific ghost type by index
    # (In the standard game, ghosts are in a specific order)
    ghost_index = self.game.ghosts.index(ghost) if ghost in self.game.ghosts else -1
    
    # Default weight if we can't identify
    if ghost_index == -1:
        return 1.0
        
    # Weights based on ghost behavior patterns
    if ghost_index == 0:  # Blinky (Red) - Most aggressive
        return 1.5
    elif ghost_index == 1:  # Pinky (Pink) - Ambusher
        return 1.3
    elif ghost_index == 2:  # Inky (Cyan) - Complex targeting
        return 1.2
    elif ghost_index == 3:  # Clyde (Orange) - Less aggressive
        return 0.8
    else:
        return 1.0

def calculate_reward(self):
    """Enhanced reward function that considers safety and predictions"""
    current_score = self.game.score
    current_pellets = len(self.game.pellets.pelletList)
    current_powerpellets = len(self.game.pellets.powerpellets)
    current_lives = self.game.lives
    
    reward = 0
    
    # Significant penalty for losing a life
    reward -= 100 * (self.previous_lives - current_lives)
    
    # Small penalty to encourage faster completion
    reward -= 0.01
    
    # Reward for eating pellets
    reward += 5 * (self.previous_pellets - current_pellets)
    
    # Reward for score increase
    reward += 0.1 * (current_score - self.previous_score)
    
    # Reward for eating a power pellet
    if current_powerpellets < self.previous_powerpellets:
        reward += 50
    
    # Reward for eating a frightened ghost
    if self.game.ghost_eaten_in_last_step:
        reward += 75
    
    # Safety-aware rewards
    # Calculate safety at current state
    safety_score = self.calculate_safety_map(self.get_state())
    
    # Reward for staying in safe positions (more escape routes)
    reward += 0.2 * safety_score
    
    # Penalty for being near predicted future ghost positions
    predicted_positions = self.predict_ghost_positions(steps_ahead=3)
    if predicted_positions:
        pacman_pos = (self.game.pacman.position.x, self.game.pacman.position.y)
        
        # Check collision risk for the next few steps
        collision_risk = 0
        for step, positions in enumerate(predicted_positions):
            # Earlier steps are more important
            step_weight = 1.0 / (step + 1)
            
            for i, pos in enumerate(positions):
                if pos is None:  # Skip frightened ghosts
                    continue
                
                # Calculate distance to predicted ghost position
                ghost_distance = ((pos[0] - pacman_pos[0])**2 + (pos[1] - pacman_pos[1])**2)**0.5
                
                # If very close to predicted position, add collision risk
                if ghost_distance < 30:  # Danger radius
                    ghost_weight = self.get_ghost_danger_weight(self.game.ghosts[i])
                    collision_risk += ghost_weight * step_weight * (30 - ghost_distance) / 30
        
        # Apply penalty for collision risk
        reward -= 10 * collision_risk
    
    # Reward for completing level
    if self.game.pellets.isEmpty():
        reward += 200
    
    # Update previous values
    self.previous_score = current_score
    self.previous_pellets = current_pellets
    self.previous_lives = current_lives
    self.previous_powerpellets = current_powerpellets
    
    return reward


def normalize(values):
    """Normalize a numpy array."""
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / (std + 1e-8)

class PPOAgent:
    """Proximal Policy Optimization agent with Actor-Critic architecture."""
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01, lam=0.95):
        # --- Hyperparameters ---
        # Dimension of the input state features
        self.state_size = state_size
        # Number of discrete actions
        self.action_size = action_size
        # Learning rate for optimizers
        self.learning_rate = learning_rate
        # Discount factor for future rewards
        self.gamma = gamma
        # PPO clipping parameter
        self.clip_epsilon = clip_epsilon
        # Weight for the value loss component
        self.value_coeff = value_coeff
        # Weight for the entropy bonus component
        self.entropy_coeff = entropy_coeff
        # Lambda parameter for Generalized Advantage Estimation (GAE)
        self.lam = lam

        # Actor network (policy)
        self.actor = self._build_network(output_size=action_size, activation='softmax', name='actor')
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Critic network (value function)
        self.critic = self._build_network(output_size=1, activation=None, name='critic')
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Experience buffer (stores transitions for one update cycle)
        self.buffer_states = []      # State features
        self.buffer_actions = []     # Action indices taken
        self.buffer_rewards = []     # Rewards received
        self.buffer_log_probs = []   # Log probability of the action taken
        self.buffer_values = []      # Value estimate from the critic (for state s)
        self.buffer_next_states = [] # Next state features (needed for GAE)
        self.buffer_dones = []       # Done flags (needed for GAE)

    def _build_network(self, output_size, activation, hidden_units=[128, 128], name=None):
        """Builds a simple feedforward neural network (used for both actor and critic)."""
        input_layer = tf.keras.layers.Input(shape=(self.state_size,), name=f'{name}_input')
        x = input_layer
        # Hidden layers with ReLU activation and Batch Normalization
        for i, units in enumerate(hidden_units):
             x = tf.keras.layers.Dense(units, activation='relu', name=f'{name}_dense_{i}')(x)
             # Batch Normalization helps stabilize training
             x = tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        # Output layer with specified activation (softmax for actor, linear for critic)
        output_layer = tf.keras.layers.Dense(output_size, activation=activation, name=f'{name}_output')(x)
        return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=name)


    def choose_action(self, state_features):
        """
        Chooses an action based on the current policy network, applying valid move masking.

        Args:
            state_features (np.array): The numerical feature vector representing the current state.

        Returns:
            tuple: (action_index, log_prob, value)
                   - action_index: The integer index of the chosen action.
                   - log_prob: The log probability of choosing that action under the current policy.
                   - value: The value estimate V(s) for the current state from the critic.
        """
        # Convert state features to a TensorFlow tensor
        state_tensor = tf.convert_to_tensor([state_features], dtype=tf.float32)

        # Get policy (action probabilities) and value estimate from networks
        policy = self.actor(state_tensor).numpy()[0]
        value = self.critic(state_tensor).numpy()[0, 0]

        # Masking using valid_moves included in state_features
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
                # Critical fallback: No valid moves identified! Default to action 0 (UP)
                print("CRITICAL WARNING: No valid actions available! Defaulting to action 0.")
                final_policy = np.zeros_like(policy)
                final_policy[0] = 1.0

        # Sample action from the final policy
        try:
            action = np.random.choice(self.action_size, p=final_policy)
        except ValueError as e:
             # Handle potential errors during sampling
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

        # Calculate the log probability of the chosen action
        log_prob = np.log(final_policy[action] + 1e-10)

        return action, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, next_state, done):
        """Stores a single transition step in the agent's buffer."""
        # Append all parts of the transition needed for GAE calculation and PPO updates
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)
        self.buffer_next_states.append(next_state)
        self.buffer_dones.append(done)

    def learn(self, batch_size=64, epochs=10):
        """
        Performs the PPO learning update using the transitions stored in the buffer.
        Calculates GAE and updates Actor and Critic networks over multiple epochs.

        Args:
            batch_size (int): Size of mini-batches used within each epoch.
            epochs (int): Number of times to iterate over the collected buffer data.
        """
        if not self.buffer_states: # Check if buffer is empty
            return # Nothing to learn

        # Prepare data from buffer
        # Convert lists stored in the buffer to numpy arrays for efficient processing
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

        # Iterate multiple times over the same batch
        for _ in range(epochs):
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
        """
        Computes Generalized Advantage Estimation (GAE) for a batch of transitions.
        GAE balances bias and variance in advantage estimation.

        Args:
            rewards (np.array): Array of rewards received at each step.
            values (np.array): Array of value estimates V(s_t) for each state.
            dones (np.array): Array of done flags (0.0 or 1.0) for each step.

        Returns:
            np.array: The calculated GAE advantages for each step.
        """
        batch_size = len(rewards)
        advantages = np.zeros(batch_size, dtype=np.float32)
        last_advantage = 0.0

        # Values array includes the bootstrapped value V(s_T) at index batch_size
        for t in reversed(range(batch_size)):
            mask = 1.0 - dones[t] # 0 if done, 1 if not done
            # Calculate TD error (delta)
            # delta_t = r_t + gamma * V(s_{t+1}) * mask - V(s_t)
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            # Calculate GAE using the recursive formula
            # A(s_t, a_t) = delta_t + gamma * lambda * mask * A(s_{t+1}, a_{t+1})
            advantages[t] = last_advantage = delta + self.gamma * self.lam * mask * last_advantage

        return advantages

    @tf.function # Decorator compiles this function into a TensorFlow graph for performance
    def _train_step(self, states, actions, advantages, log_probs_old, returns):
        """
        Performs a single gradient update for Actor and Critic networks using TensorFlow.

        Args:
            states: Batch of state features.
            actions: Batch of actions taken.
            advantages: Batch of calculated GAE advantages.
            log_probs_old: Batch of log probabilities under the policy that generated the actions.
            returns: Batch of calculated returns (targets for the critic).
        """
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
    """
    Main training loop for the PPO agent on the PacmanWrapper environment.

    Args:
        agent (PPOAgent): The PPO agent instance.
        env (PacmanWrapper): The Pac-Man environment wrapper.
        episodes (int): Total number of training episodes.
        max_steps (int): Maximum number of steps allowed per episode.
        update_frequency (int): Number of steps to collect before running agent.learn().
        epochs_per_update (int): Number of training epochs over the buffer data during agent.learn().
        save_interval (int): Frequency (in episodes) for saving model checkpoints.
    """
    # Training & Evaluation Metrics
    episode_rewards = []      # Stores total reward for each episode
    moving_avg_reward = []    # Stores moving average of episode rewards

    episode_steps_list = []   # Stores number of steps survived in each episode (proxy for time alive)
    moving_avg_steps = []     # Stores moving average of episode steps

    best_avg_steps = 0        # Tracks the best moving average steps achieved (for saving best model)
    global_step = 0           # Counter for total steps taken across all episodes
    avg_window = 100          # Window size for calculating moving averages

    print(f"Starting PPO Training (Survival Focus): Episodes={episodes}, MaxSteps/Ep={max_steps}, UpdateFreq={update_frequency}, Epochs/Update={epochs_per_update}")
    print(f"Saving best model based on {avg_window}-episode average time alive (steps)")

    # --- Episode Loop ---
    for episode in range(episodes):
        # Reset environment and get initial state dictionary
        state_dict = env.reset()
        # Extract features from the state dictionary
        state_features = env.extract_features(state_dict)

        # Initialize episode variables
        episode_reward = 0
        episode_steps = 0
        done = False

        # --- Step Loop (within an episode) ---
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
                print(f"--- Learning at Global Step: {global_step} ---")
                agent.learn(batch_size=update_frequency // 4, epochs=epochs_per_update)

        # --- End of Step Loop ---

        # --- End of Episode ---
        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps) # Record episode steps (time alive)

        # Calculate moving averages for all metrics
        current_avg_reward = np.mean(episode_rewards[-avg_window:]) if len(episode_rewards) >= avg_window else np.mean(episode_rewards)
        moving_avg_reward.append(current_avg_reward)

        current_avg_steps = np.mean(episode_steps_list[-avg_window:]) if len(episode_steps_list) >= avg_window else np.mean(episode_steps_list)
        moving_avg_steps.append(current_avg_steps)

        # Save best model based on moving average steps (time alive)
        # Only evaluate after enough episodes for the moving average to be meaningful
        if len(episode_steps_list) > avg_window // 2 and current_avg_steps > best_avg_steps:
            best_avg_steps = current_avg_steps
            print(f"*** New Best Average Time Alive ({avg_window} ep): {best_avg_steps:.2f} steps at episode {episode+1} ***")
            agent.save("best_survival_actor.keras", "best_survival_critic.keras")

        # Progress reporting
        if (episode + 1) % 10 == 0 or episode == 0: # Report every 10 episodes
             print(f"Ep {episode + 1} | "
                   f"Raw R: {episode_reward:.1f} | "
                   f"Avg R ({avg_window}): {current_avg_reward:.1f} | "
                   f"Steps (Time): {episode_steps} | " # Raw steps for the episode
                   f"Avg Steps ({avg_window}): {current_avg_steps:.1f} | " # Avg steps (time alive)
                   f"Total Steps: {global_step}")

        # Checkpoint saving
        if (episode + 1) % save_interval == 0:
             print(f"--- Saving checkpoint at episode {episode + 1} ---")
             agent.save(f"checkpoint_{episode + 1}_actor.keras",
                        f"checkpoint_{episode + 1}_critic.keras")

    # --- End of Training Loop ---

    # Final save and plot
    print("--- Training finished. Saving final model and plots. ---")
    agent.save("final_survival_actor.keras", "final_survival_critic.keras")

    plt.figure(figsize=(12, 10))

    # Plot Average Reward
    plt.subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
    plt.plot(moving_avg_reward, linewidth=2, label=f'{avg_window}-Episode Avg Reward', color='blue')
    plt.ylabel("Average Reward")
    plt.title("PPO Training Progress (Reward)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Average Time Alive (Steps)
    plt.subplot(2, 1, 2) # 2nd plot
    plt.plot(moving_avg_steps, linewidth=2, label=f'{avg_window}-Episode Avg Time Alive (Steps)', color='green')
    plt.ylabel("Average Steps per Episode")
    plt.title("PPO Training Progress (Time Alive)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("ppo_survival_training_progress.png")
    plt.close()
    print("Survival training progress plot saved to ppo_survival_training_progress.png")

    return agent


if __name__ == "__main__":
    # Configuration
    episodes = 1000                 # Total number of episodes to train
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
    ppo_agent = PPOAgent(state_size, action_size, learning_rate=0.0001, clip_epsilon=0.2, entropy_coeff=0.01)

    # Load pre-trained model if available
    try:
        ppo_agent.load("final_survival_actor.keras", "final_survival_critic.keras")
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