import pygame
from pygame.locals import *
import numpy as np
import random
import time
import pickle
import collections
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from constants import *
from run import GameController

from ppo import PPOAgent, PacmanWrapper as PPOPacmanWrapper
from Dora_The_Explorer import SimpleRLAgent

# Direction conversion mappings
DIRECTION_TO_INDEX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
INDEX_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_INDEX.items()}
ACTION_DIM = 4

class SurvivalBasedPacmanWrapper:

    def __init__(self):
        # Initializing game controller
        self.game = GameController()
        # Starting the game logic
        self.game.startGame()
        
        # Initializing state variables for reward calculation
        # Storing previous state values to calculate changes
        self.previous_score = 0
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        # Tracking if Pacman was alive in the previous step
        self.pacman_was_alive = True
        
        # Exploration bookkeeping
        # Counts how many times each state (position) has been visited (for novelty reward)
        self.visit_counts = collections.Counter()
        # Stores the last loop window positions to detect back-and-forth movement
        self.recent_positions = collections.deque(maxlen=8)
        # Stores the previous action taken (direction) for reverse/corridor checks
        self.prev_direction = STOP
        
        # Start the game by unpausing
        self.unpause_game()
    
    def unpause_game(self):
        """
        Unpause the game by simulating key presses (SPACE or RETURN).
        """
        for _ in range(3):
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
            time.sleep(0.05)
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
            time.sleep(0.2)
            # Try to unpause repeatedly after death
            for _ in range(5):
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
                time.sleep(0.05)
                self.game.update()
                # If game unpaused, stop
                if not self.game.pause.paused:
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
                'direction': pacman.direction,
                'valid_moves': [],
                'ghosts': [],
                'score': self.game.score,
                'lives': self.game.lives,
                'pellet_here': False
            }
        
        # Pacman's current position
        position = (int(pacman.position.x), int(pacman.position.y))
        # Get valid directions Pacman can move into from the current node
        valid_moves = pacman.validDirections()
        
        # Information about each ghost
        ghosts = []
        for ghost in self.game.ghosts:
            ghost_info = {
                'position': (int(ghost.position.x), int(ghost.position.y)),
                'frightened': ghost.mode.current is FREIGHT
            }
            ghosts.append(ghost_info)
        
        # Check if there is a pellet at Pacman's current exact position
        pellet_here = False
        # Iterate through the visible pellets
        for p in self.game.pellets.pelletList:
            if int(p.position.x) == position[0] and int(p.position.y) == position[1]:
                pellet_here = True
                break
        
        # Return the state dictionary
        return {
            'position': position,
            'direction': pacman.direction,
            'valid_moves': valid_moves,
            'ghosts': ghosts,
            'score': self.game.score,
            'lives': self.game.lives,
            'pellet_here': pellet_here
        }
    
    def take_action(self, action):
        """
        Executes a given direction action in the game.

        Args:
            action: The direction constant (UP, DOWN, LEFT, RIGHT) chosen by the agent.

        Returns:
            tuple: (next_state_dict, reward, done)
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()

        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()
        
        pacman = self.game.pacman
        # Allow movement only when Pac‑Man is at a node and idle
        if pacman.node is not None and pacman.target is None:
            # Check if the chosen action leads to a valid neighbor
            next_node = pacman.node.neighbors.get(action)
            if next_node:
                # Set the new direction and target node
                pacman.direction = action
                pacman.target = next_node
        
        # Update game step
        self.game.update()
        # Finalize move when target reached
        if pacman.target and pacman.position == pacman.target.position:
            pacman.node = pacman.target
            pacman.target = None
        
        # Calculate reward
        reward = self.calculate_reward(action)

        # Get next state
        next_state = self.get_state()

        # Check if episode is done
        done = self.game.lives <= 0 or self.game.pellets.isEmpty()
        
        return next_state, reward, done
    
    def calculate_reward(self, action):
        """
        Calculates reward considering score, pellets, lives, level completion,
        exploration bonus, loop penalty, and reverse move penalty.
        """
        # Get current game state values
        current_score = self.game.score
        current_pellets = len(self.game.pellets.pelletList)
        current_lives = self.game.lives
        
        # Get current position
        pac_pos = (int(self.game.pacman.position.x), int(self.game.pacman.position.y))
        
        reward = 0
        # Reward for score increase
        reward += 0.1 * (current_score - self.previous_score)
        # Reward for eating pellets
        pellets_eaten = self.previous_pellets - current_pellets

        reward += 14 * pellets_eaten
        
        # Corridor Bonus: Reward for continuing straight after eating a pellet
        if pellets_eaten > 0 and action == self.prev_direction and action != STOP:
            reward += 2
        
        # Penalty for losing lives
        lives_lost = self.previous_lives - current_lives
        reward -= 100 * lives_lost
        
        # Reward for completing level
        if self.game.pellets.isEmpty():
            reward += 500
        
        # Small penalty to encourage faster completion
        reward -= 0.1
        
        # Intrinsic Exploration Bonus (Novelty): Reward based on visit counts
        self.visit_counts[pac_pos] += 1
        if any(int(p.position.x) == pac_pos[0] and int(p.position.y) == pac_pos[1]
               for p in self.game.pellets.pelletList):
            reward += 0.75 / self.visit_counts[pac_pos]

        
        # Loop Deterrent Penalty: Penalize if Pacman bounces back and forth between few spots
        if len(self.recent_positions) >= 8 and len(set(self.recent_positions)) <= 2:
            reward -= 2.0
        self.recent_positions.append(pac_pos)
        
        # Reverse Move Penalty: Penalize for immediately reversing direction
        reversal_map = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action == reversal_map.get(self.prev_direction, STOP):
            reward -= 0.5
        self.prev_direction = action
        
        # Update previous values
        self.previous_score = current_score
        self.previous_pellets = current_pellets
        self.previous_lives = current_lives
        
        # Add reward for finishing above 75% of the level
        if (240 - current_pellets) / 240 > 0.75 and self.previous_lives > current_lives:
            reward += 100

        return reward
    
    def reset(self):
        """
        Reset the game for a new episode or next level.
        """
        # Check if level is complete
        if self.game.pellets.isEmpty():
            # Wait for level transition animation to complete
            time.sleep(1.0)
            # Force game update to process level transition
            for _ in range(20):
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
                self.game.update()
                time.sleep(0.05)
        
        # Reset game if needed (game over)
        if self.game.lives <= 0:
            self.game.restartGame()
        
        # Make sure game is unpaused
        self.unpause_game()
        
        # Reset Pac-Man state
        pacman = self.game.pacman
        pacman.setPosition()
        pacman.direction = STOP
        pacman.target = None
        
        # Reset tracking variables
        self.previous_score = self.game.score
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        
        # Clear exploration tracking variables
        self.visit_counts.clear()
        self.recent_positions.clear()
        self.prev_direction = STOP
        
        return self.get_state()

    def get_metrics(self):
        """
        Get current gameplay metrics for analysis.
        """
        metrics = {
            'score': self.game.score,
            'level': self.game.level,
            'lives': self.game.lives,
            'remaining_pellets': len(self.game.pellets.pelletList),
            'total_pellets': self.previous_pellets + (self.game.level * 240),  # Estimate of original pellet count
            'pellets_eaten': 240 - len(self.game.pellets.pelletList)  # Assuming 240 pellets per level
        }
        
        # Calculate percentage of level completed
        if metrics['total_pellets'] > 0:
            metrics['completion_percentage'] = (metrics['pellets_eaten'] / 240) * 100
        else:
            metrics['completion_percentage'] = 0
            
        return metrics

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


class SurvivalBasedAgent:
    """
    Hybrid agent switching between PPO (survival) and Q-learning (exploration)
    based on a calculated threat value from nearby ghosts.
    """
    def __init__(self, state_size, action_size):
        # Initialize the PPO agent (handles survival situations)
        self.ppo_agent = PPOAgent(state_size, action_size)
        # Initialize the Q-learning agent (handles exploration/pellet collection)
        self.q_agent = SimpleRLAgent()
        # Threat value above which the PPO agent takes control
        self.threat_threshold = 0.05
        # Weights for calculating threat (higher weight = more threatening ghost)
        self.ghost_weights = [2.0, 1.6, 1.2, 1.0]
        # Offset added to distance in threat calculation (prevents division by zero)
        self.min_distance_offset = 20.0
        # Debug flag
        self.debug = False
        # Counters for actions taken by each agent type per episode
        self.survival_mode_count = 0
        self.explore_mode_count = 0
        self.episode_steps = 0
        self.episode_rewards = 0
        # Track previous mode to only print when there's a change
        self.previous_mode_was_survival = None
    
    def calculate_threat_value(self, state_dict):
        """Estimates threat level using weighted Manhattan distances to ghosts."""
        # Pacman's current position
        pacman_pos = np.array(state_dict['position'])
        # Initialize threat
        threat_value = 0.0
        # Iterate through ghosts, calculate contribution to threat
        for i, ghost in enumerate(state_dict['ghosts']):
            # Ignore frightened ghosts
            if ghost['frightened']:
                continue
                
            ghost_pos = np.array(ghost['position'])
            
            # Calculate Manhattan distance (|dx| + |dy|)
            distance = np.sum(np.abs(pacman_pos - ghost_pos))
            # Get weight for this ghost type
            ghost_weight = self.ghost_weights[i] if i < len(self.ghost_weights) else 0.5
            # Add weighted inverse distance to total threat
            threat_value += ghost_weight / (distance + self.min_distance_offset)
        
        return threat_value
    
    def choose_action(self, state_dict):
        """Selects action: PPO if threat high, Q-agent if threat low."""
        # Features are extracted only if PPO is chosen
        state_features = None
        # Calculate current threat level
        threat_value = self.calculate_threat_value(state_dict)
        
        # If threat is high -> Use PPO
        if threat_value > self.threat_threshold:
            # Extract features needed for PPO network input
            if state_features is None:
                state_features = env.extract_features(state_dict)
            
            # Get action index from PPO agent
            action_index, _, _ = self.ppo_agent.choose_action(state_features)
            # Convert index back to game direction constant
            action = INDEX_TO_DIRECTION[action_index]
            
            # Track PPO action
            self.survival_mode_count += 1
            # Set mode flag
            survival_mode = True
            
            # Only print when switching from exploration to survival
            if self.previous_mode_was_survival is False:
                print(f"🚨 SURVIVAL MODE! Threat: {threat_value:.4f} > {self.threat_threshold}")
                # Debug print of ghost distances
                if self.debug:
                    pacman_pos = np.array(state_dict['position'])
                    for i, ghost in enumerate(state_dict['ghosts']):
                        if not ghost['frightened']:
                            ghost_pos = np.array(ghost['position'])
                            distance = np.sum(np.abs(pacman_pos - ghost_pos))
                            print(f"    Ghost {i}: Manhattan distance = {distance:.1f}, weight = {self.ghost_weights[i]}")
        # If threat is low -> Use Q-Agent
        else:
            # Get action from Q-learning agent
            action = self.q_agent.choose_action(state_dict)
            # Track Q-agent action
            self.explore_mode_count += 1
            # Set mode flag
            survival_mode = False
            
            # Only print when switching from survival to exploration
            if self.previous_mode_was_survival is True:
                print(f"🔍 EXPLORATION MODE: Threat: {threat_value:.4f} <= {self.threat_threshold}")
        
        # Update previous mode for next comparison
        self.previous_mode_was_survival = survival_mode
        
        return action, survival_mode, threat_value
    
    def learn(self, state, action, reward, next_state, done, survival_mode):
        """Updates either the PPO agent's buffer/networks or the Q-agent's Q-table."""
        # If survival mode was active (PPO chose the action)
        if survival_mode:
            # Convert game action constant back to numerical index for PPO
            action_index = DIRECTION_TO_INDEX.get(action, 0)
            # Extract features for current and next states
            state_features = env.extract_features(state)
            next_state_features = env.extract_features(next_state)
            
            # Get log_prob and value estimate for the *current* state (needed for PPO buffer)
            _, log_prob, value = self.ppo_agent.choose_action(state_features)
            
            # Store the complete transition information in PPO agent's buffer
            self.ppo_agent.store_transition(
                state=state_features,
                action=action_index,
                reward=reward,
                log_prob=log_prob,
                value=value,
                next_state=next_state_features,
                done=done
            )
            
            # Check if the PPO buffer is full enough to trigger a learning update
            if len(self.ppo_agent.buffer_states) >= 2048:
                # Call PPO agent's learn method
                self.ppo_agent.learn(batch_size=512, epochs=10)
        # If exploration mode was active (Q-agent chose the action)
        else:
            # Call Q-agent's learn method
            self.q_agent.learn(state, action, reward, next_state, done)
    
    def save_agents(self, ppo_actor_path="survival_ppo_actor.keras", ppo_critic_path="survival_ppo_critic.keras", q_agent_path="survival_q_agent.pkl"):
        """Saves PPO model files (.keras) and Q-agent state (.pkl)."""
        # Save PPO actor and critic networks
        self.ppo_agent.save(ppo_actor_path, ppo_critic_path)
        # Save Q-agent's Q-table and other relevant data
        self.q_agent.save(q_agent_path)
        print(f"Agents saved to {ppo_actor_path}, {ppo_critic_path}, and {q_agent_path}")


def train(episodes=1000, max_steps=3000, save_interval=100, fast_mode=True):
    """Main training loop for the hybrid SurvivalBasedAgent."""
    global env
    # --- Setup Pygame Display/Mode ---
    # Disable or reduce rendering if in fast mode
    if fast_mode:
        # Set pygame to run headless if possible
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Try to set a smaller screen size to reduce rendering load
        pygame.display.set_mode((1, 1))
    else:
        # If fast_mode is disabled, ensure the display is properly initialized
        pygame.display.set_mode((800, 600))  # Use normal game window size

    # --- Initialize Environment and Agent ---
    env = SurvivalBasedPacmanWrapper()
    # Get initial state to determine feature vector size
    initial_state = env.reset()
    state_features = env.extract_features(initial_state)
    state_size = len(state_features)
    action_size = ACTION_DIM
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Training with {'fast' if fast_mode else 'visual'} mode")
    
    # Create the hybrid agent instance
    agent = SurvivalBasedAgent(state_size, action_size)
    print("Training from scratch — no pre-trained models loaded.")
    # Ensure checkpoint directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    # --- Setup Checkpoints and Metrics ---
    all_rewards = []              # List of total rewards per episode
    all_survival_ratio = []       # List of survival mode ratios per episode
    moving_avg_rewards = []       # List of moving average rewards
    episode_threat_values = []    # List of lists storing threat values per step per episode
    completion_rate_per_100 = []
    completion_percentages = []
    
    # --- Timing ---
    # Track total training time
    start_time = time.time()
    # Track total steps across all episodes
    steps_taken = 0
    
    # --- Main Training Loop ---
    for episode in range(episodes):
        # Time the duration of this episode
        episode_start_time = time.time()
        # Reset env for new episode
        state = env.reset()
        # Reset episode metrics
        total_reward = 0
        steps = 0
        # Reset agent's episode counters
        agent.survival_mode_count = 0
        agent.explore_mode_count = 0
        # Reset mode tracking at start of episode
        agent.previous_mode_was_survival = None
        episode_survival_ratio = 0
        # Create list for this episode's threat values
        episode_threat_values.append([])
        
        # Episode termination flag
        done = False
        # --- Step Loop (within episode) ---
        while not done and steps < max_steps:
            # Agent chooses action (hybrid logic selects PPO or Q)
            action, survival_mode, threat_value = agent.choose_action(state)
            # Store threat value for analysis
            episode_threat_values[-1].append(threat_value)
            # Environment processes action
            next_state, reward, done = env.take_action(action)
            # Agent learns from the transition (updates PPO buffer or Q-table)
            agent.learn(state, action, reward, next_state, done, survival_mode)
            
            # Update state and counters for next step
            state = next_state
            total_reward += reward
            steps += 1
            steps_taken += 1
            
            # Add a small delay if in visual mode to make the game visible
            if not fast_mode and steps % 5 == 0:  # Only delay every 5 steps for smoother visualization
                pygame.display.update()  # Update the display
                time.sleep(0.01)  # Short delay to make game visible but not too slow
        
        # --- End of Episode ---
        # Calculate episode survival ratio
        total_actions = agent.survival_mode_count + agent.explore_mode_count
        if total_actions > 0:
            episode_survival_ratio = agent.survival_mode_count / total_actions
        
        # Store episode metrics
        all_rewards.append(total_reward)
        all_survival_ratio.append(episode_survival_ratio)
        
        metrics = env.get_metrics()
        completion_pct = metrics.get("completion_percentage", 0)
        completion_percentages.append(completion_pct)
        
        # Calculate moving average reward
        window_size = min(100, len(all_rewards))
        moving_avg = sum(all_rewards[-window_size:]) / window_size
        moving_avg_rewards.append(moving_avg)
        
        # --- Performance Metrics ---
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        steps_per_second = steps / max(episode_time, 0.1)
        avg_steps_per_second = steps_taken / max(total_time, 0.1)
        
        # --- Print Episode Summary ---
        print(f"Episode {episode + 1}/{episodes} | Score: {state['score']} | "
              f"Reward: {total_reward:.2f} | Avg(100): {moving_avg:.2f} | "
              f"Steps: {steps} | Time: {episode_time:.1f}s | Speed: {steps_per_second:.1f} steps/s | "
              f"Survival ratio: {episode_survival_ratio:.2f} | Completion: {completion_pct:.1f}%")
        
        # --- Save Checkpoint Models and Plots ---
        if (episode + 1) % save_interval == 0 or episode == episodes - 1:
            agent.save_agents(
                ppo_actor_path=f"checkpoints/survival_ppo_actor_ep{episode + 1}.keras",
                ppo_critic_path=f"checkpoints/survival_ppo_critic_ep{episode + 1}.keras",
                q_agent_path=f"checkpoints/survival_q_agent_ep{episode + 1}.pkl"
            )

            avg_completion_100 = np.mean(completion_percentages[-100:])
            completion_rate_per_100.append(avg_completion_100)

            plot_training_progress(
                all_rewards, 
                moving_avg_rewards, 
                all_survival_ratio,
                completion_percentages,
                completion_rate_per_100,
                episode + 1
            )

    
    # --- End of Training Loop ---
    # Save the final agent models after training completes
    agent.save_agents()
    
    # Generate final summary plots
    plot_training_progress(
        all_rewards, 
        moving_avg_rewards, 
        all_survival_ratio,
        completion_percentages,
        completion_rate_per_100,
        episodes,
        final=True
    )
    
    # --- Print Final Training Summary ---
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.0f} seconds")
    print(f"Total steps: {steps_taken}")
    print(f"Average speed: {steps_taken/total_time:.1f} steps/second")
    
    # Return the trained hybrid agent
    return agent


def plot_training_progress(rewards, moving_avg, survival_ratio, completion_percentages, completion_rate_per_100, episode, final=False):
    """Generates and saves plots for reward and survival mode ratio during training."""
    plt.figure(figsize=(12, 10))
    
    # --- Subplot 1: Rewards ---
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward', color='lightblue')
    plt.plot(moving_avg, linewidth=2, label='Moving Average (100 ep)', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress - Rewards')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Subplot 2: Survival Mode Ratio ---
    plt.subplot(2, 1, 2)
    plt.plot(survival_ratio, linewidth=2, label='Survival Mode Ratio', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Survival Mode Ratio')
    plt.title('Proportion of Actions Using Survival (PPO) Agent')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(completion_rate_per_100)), completion_rate_per_100, 'mo-', linewidth=2)
    plt.xlabel('Checkpoint (x100 Episodes)')
    plt.ylabel('Avg Completion %')
    plt.title('Average Completion Rate Every 100 Episodes')
    plt.grid(True, linestyle='--', alpha=0.6)

    if final:
        plt.savefig(f"survival_based_training_completion_rate_final.png")
    else:
        plt.savefig(f"checkpoints/survival_based_training_completion_rate_ep{episode}.png")

    plt.close()

    plt.tight_layout() # Adjust spacing
    
    # Determine filename for saving the plot
    if final:
        plt.savefig(f"survival_based_training_final.png")
    else:
        plt.savefig(f"checkpoints/survival_based_training_ep{episode}.png")
    
    plt.close()

def evaluate_with_analytics(agent, episodes=10, render=True, fast_mode=False):
    """
    Evaluate the trained agent with detailed analytics metrics.
    
    Args:
        agent: The trained agent
        episodes: Number of episodes to evaluate
        render: Whether to render the game
        fast_mode: Whether to use headless mode
    
    Returns:
        dict: Comprehensive performance metrics and analytics
    """
    global env
    
    # Reset environment for evaluation
    if fast_mode:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.set_mode((1, 1))
    elif render:
        pygame.display.set_mode((800, 600))
    
    env = SurvivalBasedPacmanWrapper()
    
    # Metrics to track
    scores = []
    steps_list = []
    levels_completed = 0
    survival_ratios = []
    completion_percentages = []
    
    # Detailed analytics
    pellet_collection_rates = []  # Pellets per step
    ghost_proximities = []  # Average threat values
    mode_switches_per_episode = []
    
    # Action distributions
    action_distributions = {
        'overall': {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0},
        'survival': {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0},
        'exploration': {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
    }
    
    print(f"\nEvaluating agent with analytics for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_threats = []
        
        # Reset counters
        agent.survival_mode_count = 0
        agent.explore_mode_count = 0
        agent.previous_mode_was_survival = None
        mode_switches = 0
        current_mode = None
        
        # Track detailed data
        initial_pellets = len(env.game.pellets.pelletList)
        pellets_collected = 0
        
        done = False
        while not done and steps < 5000:  # Longer step limit for evaluation
            # Choose action but don't learn
            action, survival_mode, threat_value = agent.choose_action(state)
            episode_threats.append(threat_value)
            
            # Track mode switches
            if current_mode is not None and current_mode != survival_mode:
                mode_switches += 1
            current_mode = survival_mode
            
            # Track action distributions
            action_distributions['overall'][action] += 1
            if survival_mode:
                action_distributions['survival'][action] += 1
            else:
                action_distributions['exploration'][action] += 1
            
            # Take action
            next_state, reward, done = env.take_action(action)
            
            # Track pellet collection
            pellets_now = len(env.game.pellets.pelletList)
            pellets_collected += (env.previous_pellets - pellets_now)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update display if rendering
            if render and not fast_mode and steps % 3 == 0:
                pygame.display.update()
                time.sleep(0.01)
        
        # Get final metrics
        metrics = env.get_metrics()
        
        # Calculate survival ratio
        total_actions = agent.survival_mode_count + agent.explore_mode_count
        if total_actions > 0:
            survival_ratio = agent.survival_mode_count / total_actions
            survival_ratios.append(survival_ratio)
        
        # Check if level was completed
        level_completed = env.game.pellets.isEmpty()
        levels_completed += (1 if level_completed else 0)
        
        # Record metrics
        scores.append(metrics['score'])
        steps_list.append(steps)
        completion_percentages.append(metrics['completion_percentage'])
        
        # Record detailed analytics
        if steps > 0:
            pellet_collection_rates.append(pellets_collected / steps)
        ghost_proximities.append(sum(episode_threats) / len(episode_threats) if episode_threats else 0)
        mode_switches_per_episode.append(mode_switches)
        
        print(f"Eval Episode {episode+1}/{episodes} | "
              f"Score: {metrics['score']} | Steps: {steps} | "
              f"Completion: {metrics['completion_percentage']:.1f}% | "
              f"{'COMPLETED' if level_completed else 'FAILED'}")

if __name__ == "__main__":
    
    # Ask user whether the user wants to see the Pac-Man window during training
    print("--------------------------------------------------------------------------------------------------------------------------")
    print("The training procedure takes a long time, if you wish to speed things up you would want to avoid using the Pac-Man window.")
    user_input = input("Do you want to see the Pac-Man window during training? (y/n): ")
    print("--------------------------------------------------------------------------------------------------------------------------")
    fast_mode = False if user_input.lower() == 'y' else True
    
    print(f"Fast mode {'disabled' if not fast_mode else 'enabled'} - game {'will' if not fast_mode else 'will not'} be visible during training")
    
    # Train with user's choice of fast_mode to control window visibility
    trained_agent = train(episodes=1000, max_steps=3000, save_interval=100, fast_mode=True)
    print("\n--------------------------------------------------------------------------------------------------------------------------")
    print("Training complete! Running comprehensive analytics on the trained agent...")
    analytics_results = evaluate_with_analytics(trained_agent, episodes=10, render=not fast_mode, fast_mode=fast_mode)

    # Generate a threshold sensitivity analysis
    print("\n--------------------------------------------------------------------------------------------------------------------------")
    print("Performing threshold sensitivity analysis...")

    # Store the original threshold
    original_threshold = trained_agent.threat_threshold

    # Test different thresholds
    thresholds = [0.01, 0.02, 0.05, 0.075, 0.1]
    threshold_results = {}

    for threshold in thresholds:
        print(f"\nTesting threshold value: {threshold}")
        # Update the threshold
        trained_agent.threat_threshold = threshold
        
        # Run a quick evaluation
        env = SurvivalBasedPacmanWrapper()  # Reset environment for consistent testing
        
        # Track metrics
        scores = []
        completion_percentages = []
        survival_ratios = []
        
        for episode in range(5):  # Quick 5-episode test for each threshold
            state = env.reset()
            steps = 0
            
            # Reset counters
            trained_agent.survival_mode_count = 0
            trained_agent.explore_mode_count = 0
            
            done = False
            while not done and steps < 3000:
                action, survival_mode, _ = trained_agent.choose_action(state)
                next_state, reward, done = env.take_action(action)
                state = next_state
                steps += 1
            
            # Calculate survival ratio
            total_actions = trained_agent.survival_mode_count + trained_agent.explore_mode_count
            if total_actions > 0:
                survival_ratio = trained_agent.survival_mode_count / total_actions
                survival_ratios.append(survival_ratio)
            
            # Get metrics
            metrics = env.get_metrics()
            scores.append(metrics['score'])
            completion_percentages.append(metrics['completion_percentage'])
            
            print(f"  Episode {episode+1}: Score={metrics['score']}, "
                f"Completion={metrics['completion_percentage']:.1f}%, "
                f"Survival Ratio={survival_ratio:.2f}")
        
        # Calculate averages
        avg_score = sum(scores) / len(scores)
        avg_completion = sum(completion_percentages) / len(completion_percentages)
        avg_survival_ratio = sum(survival_ratios) / len(survival_ratios)
        
        # Store results
        threshold_results[threshold] = {
            'avg_score': avg_score,
            'avg_completion': avg_completion,
            'avg_survival_ratio': avg_survival_ratio
        }

    # Restore original threshold
    trained_agent.threat_threshold = original_threshold

    # Plot threshold sensitivity results
    plt.figure(figsize=(15, 10))

    # Extract data for plotting
    thresholds_list = list(threshold_results.keys())
    scores_list = [threshold_results[t]['avg_score'] for t in thresholds_list]
    completion_list = [threshold_results[t]['avg_completion'] for t in thresholds_list]
    survival_list = [threshold_results[t]['avg_survival_ratio'] for t in thresholds_list]

    # Score vs Threshold
    plt.subplot(3, 1, 1)
    plt.plot(thresholds_list, scores_list, 'bo-', linewidth=2)
    plt.xlabel('Threat Threshold')
    plt.ylabel('Average Score')
    plt.title('Effect of Threat Threshold on Performance')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Completion Rate vs Threshold
    plt.subplot(3, 1, 2)
    plt.plot(thresholds_list, completion_list, 'go-', linewidth=2)
    plt.xlabel('Threat Threshold')
    plt.ylabel('Level Completion (%)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Survival Ratio vs Threshold
    plt.subplot(3, 1, 3)
    plt.plot(thresholds_list, survival_list, 'ro-', linewidth=2)
    plt.xlabel('Threat Threshold')
    plt.ylabel('Survival Mode Ratio')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('threshold_sensitivity.png')
    plt.close()

    # Find the best threshold
    best_score_threshold = thresholds_list[scores_list.index(max(scores_list))]
    best_completion_threshold = thresholds_list[completion_list.index(max(completion_list))]

    print("\nThreshold Sensitivity Analysis Results:")
    print("----------------------------------------")
    for threshold in thresholds_list:
        print(f"Threshold {threshold}:")
        print(f"  Score: {threshold_results[threshold]['avg_score']:.1f}")
        print(f"  Completion Rate: {threshold_results[threshold]['avg_completion']:.1f}%")
        print(f"  Survival Ratio: {threshold_results[threshold]['avg_survival_ratio']:.2f}")

    print(f"\nBest threshold for score: {best_score_threshold}")
    print(f"Best threshold for completion rate: {best_completion_threshold}")
    print("----------------------------------------")