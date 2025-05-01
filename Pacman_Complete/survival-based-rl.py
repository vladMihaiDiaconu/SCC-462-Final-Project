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

DIRECTION_TO_INDEX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
INDEX_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_INDEX.items()}
ACTION_DIM = 4

class SurvivalBasedPacmanWrapper:
    """
    A specialized Pac-Man game wrapper that focuses on survival metrics and efficient game control.
    This wrapper handles game state management, action execution, and reward calculation
    with a focus on encouraging Pac-Man to both survive longer and collect more pellets.
    """
    def __init__(self):
        # Initialize the game controller
        self.game = GameController()
        self.game.startGame()
        
        # Track game state variables for reward calculation
        self.previous_score = 0
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        
        # Variables to track exploration patterns and prevent loops
        self.visit_counts = collections.Counter()  # Track how often we visit each position
        self.recent_positions = collections.deque(maxlen=8)  # Track last 8 positions to detect loops
        self.prev_direction = STOP  # Track previous direction to penalize reversals
        
        # Ensure game starts unpaused
        self.unpause_game()
    
    def unpause_game(self):
        """
        Attempt to unpause the game by sending Space and Enter key events.
        Uses a reduced number of attempts and shorter delays for efficiency.
        """
        for _ in range(3):  # Try 3 times to unpause
            # Send both Space and Enter key events
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
            time.sleep(0.05)  # Short delay to let game process the key events
            self.game.update()
            
            # If successfully unpaused, exit the loop
            if not self.game.pause.paused:
                break
    
    def check_if_pacman_died(self):
        """
        Check if Pac-Man just died and handle the death by unpausing the game.
        This is crucial to prevent the game from staying paused after a death.
        """
        if self.pacman_was_alive and not self.game.pacman.alive:
            # Short delay to let death animation play
            time.sleep(0.2)
            
            # Attempt multiple times to unpause after death
            for _ in range(5):
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
                time.sleep(0.05)
                self.game.update()
                
                # If successfully unpaused, exit the loop
                if not self.game.pause.paused:
                    break
        
        # Update alive status for next check
        self.pacman_was_alive = self.game.pacman.alive
    
    def get_state(self):
        """
        Get the current game state as a dictionary containing all relevant information
        for decision making, including Pac-Man's position, valid moves, ghost information,
        and whether there's a pellet at the current position.
        """
        # First check if Pac-Man died and handle it
        self.check_if_pacman_died()
        
        # Ensure game is unpaused
        if self.game.pause.paused:
            self.unpause_game()
        
        pacman = self.game.pacman
        
        # Return minimal state if Pac-Man is not on a node
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
        
        # Get Pac-Man's position and valid moves
        position = (int(pacman.position.x), int(pacman.position.y))
        valid_moves = pacman.validDirections()
        
        # Get ghost information
        ghosts = []
        for ghost in self.game.ghosts:
            ghost_info = {
                'position': (int(ghost.position.x), int(ghost.position.y)),
                'frightened': ghost.mode.current is FREIGHT  # Track if ghost is frightened (blue)
            }
            ghosts.append(ghost_info)
        
        # Check if there's a pellet at Pac-Man's current position
        pellet_here = False
        for p in self.game.pellets.pelletList:
            if int(p.position.x) == position[0] and int(p.position.y) == position[1]:
                pellet_here = True
                break
        
        # Return complete state dictionary
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
        Execute the given action in the game and return the next state, reward, and done flag.
        Uses direct target node setting for more reliable movement control.
        """
        # First check if Pac-Man died and handle it
        self.check_if_pacman_died()
        
        # Ensure game is unpaused
        if self.game.pause.paused:
            self.unpause_game()
        
        pacman = self.game.pacman
        
        # Set target node if Pac-Man is on a node and not already moving to a target
        if pacman.node is not None and pacman.target is None:
            next_node = pacman.node.neighbors.get(action)
            if next_node:
                pacman.direction = action
                pacman.target = next_node
        
        # Update game state (minimal delay for faster learning)
        self.game.update()
        
        # Update Pac-Man's node when target is reached
        if pacman.target and pacman.position == pacman.target.position:
            pacman.node = pacman.target
            pacman.target = None
        
        # Calculate reward based on the action taken
        reward = self.calculate_reward(action)
        
        # Get the next state and check if episode is done
        next_state = self.get_state()
        done = self.game.lives <= 0 or self.game.pellets.isEmpty()
        
        return next_state, reward, done
    
    def calculate_reward(self, action):
        """
        Calculate the reward for the current step using multiple factors:
        - Score increase
        - Pellets eaten
        - Lives lost
        - Exploration patterns (penalize revisiting same spots and loops)
        - Direction changes (penalize reversals)

        This complex reward function encourages efficient pellet collection
        while discouraging repetitive movement patterns.
        """
        # Get current game state metrics
        current_score = self.game.score
        current_pellets = len(self.game.pellets.pelletList)
        current_lives = self.game.lives
        
        # Get Pac-Man's current position
        pac_pos = (int(self.game.pacman.position.x), int(self.game.pacman.position.y))
        
        # Initialize reward
        reward = 0
        
        # Reward for score increase (from ghosts, fruit, etc.)
        reward += 0.1 * (current_score - self.previous_score)
        
        # Calculate pellets eaten this step
        pellets_eaten = self.previous_pellets - current_pellets

        # Substantial reward for eating pellets
        reward += 14 * pellets_eaten
        
        # Extra reward for eating pellets while continuing in same direction
        # This encourages Pac-Man to keep moving in productive directions
        if pellets_eaten > 0 and action == self.prev_direction and action != STOP:
            reward += 2
        
        # Big penalty for losing lives
        lives_lost = self.previous_lives - current_lives
        reward -= 100 * lives_lost
        
        # Large reward for completing a level
        if self.game.pellets.isEmpty():
            reward += 500
        
        # Small penalty each step to encourage faster completion
        reward -= 0.1
        
        # Track position visits and reward exploring less-visited pellet positions
        self.visit_counts[pac_pos] += 1
        if any(int(p.position.x) == pac_pos[0] and int(p.position.y) == pac_pos[1]
               for p in self.game.pellets.pelletList):
            # Diminishing returns for revisiting the same pellet position
            reward += 0.75 / self.visit_counts[pac_pos]

        # Detect and penalize movement loops (agent moving back and forth)
        if len(self.recent_positions) >= 8 and len(set(self.recent_positions)) <= 2:
            reward -= 2.0  # Significant penalty for getting stuck in a loop
        self.recent_positions.append(pac_pos)
        
        # Penalize direction reversals to discourage zigzag movement
        reversal_map = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action == reversal_map.get(self.prev_direction, STOP):
            reward -= 0.5
        self.prev_direction = action  # Store current direction for next step
        
        # Update state variables for next step
        self.previous_score = current_score
        self.previous_pellets = current_pellets
        self.previous_lives = current_lives
        
        # Bonus reward for dying after completing most of the level
        # This encourages the agent to maximize pellets eaten even if it dies
        if (240 - current_pellets) / 240 > 0.75 and self.previous_lives > current_lives:
            reward += 100

        return reward
    
    def reset(self):
        """
        Reset the game for a new episode or next level.
        Handles level completion transitions and resets all state tracking variables.
        """
        # Check if level is complete and handle level transition
        if self.game.pellets.isEmpty():
            # Wait for level transition animation to complete
            time.sleep(1.0)
            # Force game update to process level transition
            for _ in range(20):
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
                self.game.update()
                time.sleep(0.05)
        
        # Reset game if all lives are lost
        if self.game.lives <= 0:
            self.game.restartGame()
        
        # Make sure game is unpaused
        self.unpause_game()
        
        # Reset Pac-Man's state
        pacman = self.game.pacman
        pacman.setPosition()
        pacman.direction = STOP
        pacman.target = None
        
        # Reset tracking variables for reward calculation
        self.previous_score = self.game.score
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        
        # Clear exploration tracking variables
        self.visit_counts.clear()
        self.recent_positions.clear()
        self.prev_direction = STOP
        
        # Return initial state
        return self.get_state()

    def get_metrics(self):
        """
        Get current gameplay metrics for analysis and tracking progress.
        Includes score, level, lives, and pellet collection statistics.
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
        """
        Convert the state dictionary into a normalized feature vector for neural network input.
        This transforms the complex game state into a format suitable for deep learning.
        """
        # Maximum game board dimensions for normalization
        max_x, max_y = 560, 620
        
        # Normalize Pac-Man position (0-1 range)
        pos = np.array(state_dict['position']) / np.array([max_x, max_y])
        
        # One-hot encode Pac-Man's current direction
        dir_one_hot = np.zeros(ACTION_DIM)
        if state_dict['direction'] in DIRECTION_TO_INDEX:
            dir_one_hot[DIRECTION_TO_INDEX[state_dict['direction']]] = 1
        
        # One-hot encode valid moves
        valid_moves_one_hot = np.zeros(ACTION_DIM)
        for move in state_dict['valid_moves']:
            if move in DIRECTION_TO_INDEX:
                valid_moves_one_hot[DIRECTION_TO_INDEX[move]] = 1
        
        # Process ghost information
        ghost_features = []
        num_ghosts = 4
        
        # For each ghost, include normalized position and frightened state
        for i in range(num_ghosts):
            if i < len(state_dict['ghosts']):
                ghost = state_dict['ghosts'][i]
                ghost_pos = np.array(ghost['position']) / np.array([max_x, max_y])
                frightened = np.array([float(ghost['frightened'])])
                ghost_features.extend(np.concatenate((ghost_pos, frightened)))
            else:
                # For missing ghosts, use placeholder values
                ghost_features.extend([-1.0, -1.0, -1.0])
        
        # Concatenate all features into a single vector
        features = np.concatenate((
            pos,  # Pac-Man's position (2 values)
            dir_one_hot,  # Pac-Man's direction (ACTION_DIM values)
            valid_moves_one_hot,  # Valid moves (ACTION_DIM values)
            ghost_features  # Ghost information (12 values: 3 per ghost × 4 ghosts)
        )).astype(np.float32)
        
        return features


class SurvivalBasedAgent:
    """
    A hybrid reinforcement learning agent that combines PPO and Q-learning.
    This agent switches between two modes:
    1. Survival mode (PPO): When ghosts are nearby and threatening
    2. Exploration mode (Q-learning): When it's safe to explore and collect pellets

    This approach allows specialized behavior in different game situations.
    """
    def __init__(self, state_size, action_size):
        # Initialize both RL algorithm components
        self.ppo_agent = PPOAgent(state_size, action_size)  # For survival mode
        self.q_agent = SimpleRLAgent()  # For exploration mode
        
        # Parameters for switching between modes
        self.threat_threshold = 0.05  # Threshold to switch to survival mode
        # Weights for each ghost's threat (closer ghosts have higher weights)
        self.ghost_weights = [2.0, 1.6, 1.2, 1.0]
        self.min_distance_offset = 20.0  # Prevents division by zero and adjusts sensitivity
        
        # Debugging flag
        self.debug = False
        
        # Statistics tracking
        self.survival_mode_count = 0  # Count frames in survival mode
        self.explore_mode_count = 0  # Count frames in exploration mode
        self.episode_steps = 0  # Total steps in current episode
        self.episode_rewards = 0  # Total rewards in current episode
        
        # Track previous mode to only print mode changes
        self.previous_mode_was_survival = None
    
    def calculate_threat_value(self, state_dict):
        """
        Calculate how threatening the current ghost situation is.
        Higher values indicate Pac-Man is in more danger.
        This considers ghost distances and weights closer ghosts more heavily.
        """
        pacman_pos = np.array(state_dict['position'])
        threat_value = 0.0
        
        # Assess threat from each ghost
        for i, ghost in enumerate(state_dict['ghosts']):
            # Ignore frightened (blue) ghosts as they're not threatening
            if ghost['frightened']:
                continue
                
            ghost_pos = np.array(ghost['position'])
            
            # Calculate Manhattan distance to ghost
            distance = np.sum(np.abs(pacman_pos - ghost_pos))
            
            # Apply weight based on ghost index (Blinky is usually most aggressive)
            ghost_weight = self.ghost_weights[i] if i < len(self.ghost_weights) else 0.5
            
            # Higher threat for closer ghosts, using inverse distance relationship
            threat_value += ghost_weight / (distance + self.min_distance_offset)
        
        return threat_value
    
    def choose_action(self, state_dict):
        """
        Decide which action to take based on current state.
        Dynamically switches between survival and exploration modes
        based on the current threat level.
        """
        state_features = None
        
        # Calculate how threatening the current situation is
        threat_value = self.calculate_threat_value(state_dict)
        
        if threat_value > self.threat_threshold:
            # SURVIVAL MODE: Use PPO for more sophisticated survival behavior
            if state_features is None:
                state_features = env.extract_features(state_dict)
            
            # Get action from PPO agent
            action_index, _, _ = self.ppo_agent.choose_action(state_features)
            action = INDEX_TO_DIRECTION[action_index]
            
            # Track survival mode usage
            self.survival_mode_count += 1
            survival_mode = True
            
            # Only print message when switching to survival mode
            if self.previous_mode_was_survival is False:
                print(f"🚨 SURVIVAL MODE! Threat: {threat_value:.4f} > {self.threat_threshold}")
                
                # Print detailed ghost information in debug mode
                if self.debug:
                    pacman_pos = np.array(state_dict['position'])
                    for i, ghost in enumerate(state_dict['ghosts']):
                        if not ghost['frightened']:
                            ghost_pos = np.array(ghost['position'])
                            distance = np.sum(np.abs(pacman_pos - ghost_pos))
                            print(f"    Ghost {i}: Manhattan distance = {distance:.1f}, weight = {self.ghost_weights[i]}")
        else:
            # EXPLORATION MODE: Use Q-learning for efficient pellet collection
            action = self.q_agent.choose_action(state_dict)
            
            # Track exploration mode usage
            self.explore_mode_count += 1
            survival_mode = False
            
            # Only print message when switching to exploration mode
            if self.previous_mode_was_survival is True:
                print(f"🔍 EXPLORATION MODE: Threat: {threat_value:.4f} <= {self.threat_threshold}")
        
        # Update previous mode for next comparison
        self.previous_mode_was_survival = survival_mode
        
        return action, survival_mode, threat_value
    
    def learn(self, state, action, reward, next_state, done, survival_mode):
        """
        Update the appropriate learning algorithm based on which mode was used.
        This keeps each algorithm specialized to its specific task.
        """
        if survival_mode:
            # Update PPO agent using the experience
            action_index = DIRECTION_TO_INDEX.get(action, 0)
            
            # Convert states to feature vectors for neural network
            state_features = env.extract_features(state)
            next_state_features = env.extract_features(next_state)
            
            # Get PPO-specific values
            _, log_prob, value = self.ppo_agent.choose_action(state_features)
            
            # Store transition in PPO buffer
            self.ppo_agent.store_transition(
                state=state_features,
                action=action_index,
                reward=reward,
                log_prob=log_prob,
                value=value,
                next_state=next_state_features,
                done=done
            )
            
            # Periodically update PPO network after collecting enough samples
            if len(self.ppo_agent.buffer_states) >= 2048:
                self.ppo_agent.learn(batch_size=512, epochs=10)
        else:
            # Update Q-learning agent
            self.q_agent.learn(state, action, reward, next_state, done)
    
    def save_agents(self, ppo_actor_path="survival_ppo_actor.keras", ppo_critic_path="survival_ppo_critic.keras", q_agent_path="survival_q_agent.pkl"):
        """
        Save both agents to disk for later loading.
        """
        self.ppo_agent.save(ppo_actor_path, ppo_critic_path)
        self.q_agent.save(q_agent_path)
        print(f"Agents saved to {ppo_actor_path}, {ppo_critic_path}, and {q_agent_path}")



def train(episodes=1000, max_steps=3000, save_interval=100, fast_mode=True):
    """
    Train the survival-based agent for 1000 episodes, saving at an interval of 100 episodes, limiting to 3000 steps per episode.
    The flag @fast_mode is used to increase the training speed.
    """
    global env
    
    # Disable or reduce rendering if in fast mode
    if fast_mode:
        # Set pygame to run headless if possible
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Try to set a smaller screen size to reduce rendering load
        pygame.display.set_mode((1, 1))
    else:
        # If fast_mode is disabled, ensure the display is properly initialized
        pygame.display.set_mode((800, 600))  # Use normal game window size
        
    env = SurvivalBasedPacmanWrapper()
    
    initial_state = env.reset()
    state_features = env.extract_features(initial_state)
    state_size = len(state_features)
    action_size = ACTION_DIM
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Training with {'fast' if fast_mode else 'visual'} mode")
    
    agent = SurvivalBasedAgent(state_size, action_size)
    print("Training from scratch — no pre-trained models loaded.")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    all_rewards = []
    all_survival_ratio = []
    moving_avg_rewards = []
    episode_threat_values = []
    completion_rate_per_100 = []
    completion_percentages = []
    
    # For tracking training speed
    start_time = time.time()
    steps_taken = 0
    
    # Iterate through each episode separately
    for episode in range(episodes):
        episode_start_time = time.time()
        state = env.reset()
        total_reward = 0
        steps = 0
        
        agent.survival_mode_count = 0
        agent.explore_mode_count = 0
        agent.previous_mode_was_survival = None  # Reset mode tracking at start of episode
        episode_survival_ratio = 0
        episode_threat_values.append([])

        done = False

        # Loop while within the limit of steps
        while not done and steps < max_steps:
            action, survival_mode, threat_value = agent.choose_action(state)
            
            episode_threat_values[-1].append(threat_value)
            
            next_state, reward, done = env.take_action(action)
            
            agent.learn(state, action, reward, next_state, done, survival_mode)
            
            state = next_state
            total_reward += reward
            steps += 1
            steps_taken += 1
            
            # Add a small delay if in visual mode to make the game visible
            if not fast_mode and steps % 5 == 0:  # Only delay every 5 steps for smoother visualization
                pygame.display.update()  # Update the display
                time.sleep(0.01)  # Short delay to make game visible but not too slow
        
        # All actions have to be include both single-purpose agents in the calculations
        total_actions = agent.survival_mode_count + agent.explore_mode_count
        if total_actions > 0:
            episode_survival_ratio = agent.survival_mode_count / total_actions
        
        # Retain all metrics from the episode
        all_rewards.append(total_reward)
        all_survival_ratio.append(episode_survival_ratio)
        
        metrics = env.get_metrics()
        completion_pct = metrics.get("completion_percentage", 0)
        completion_percentages.append(completion_pct)
        
        window_size = min(100, len(all_rewards))
        moving_avg = sum(all_rewards[-window_size:]) / window_size
        moving_avg_rewards.append(moving_avg)
        
        # Calculate episode speed metrics
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        steps_per_second = steps / max(episode_time, 0.1)
        avg_steps_per_second = steps_taken / max(total_time, 0.1)
        
        # Prompt the user with the episode details
        print(f"Episode {episode + 1}/{episodes} | Score: {state['score']} | "
              f"Reward: {total_reward:.2f} | Avg(100): {moving_avg:.2f} | "
              f"Steps: {steps} | Time: {episode_time:.1f}s | Speed: {steps_per_second:.1f} steps/s | "
              f"Survival ratio: {episode_survival_ratio:.2f} | Completion: {completion_pct:.1f}%")
        
        # Save the state of the agents after each episode in case of crash
        if (episode + 1) % save_interval == 0 or episode == episodes - 1:
            agent.save_agents(
                ppo_actor_path=f"checkpoints/survival_ppo_actor_ep{episode + 1}.keras",
                ppo_critic_path=f"checkpoints/survival_ppo_critic_ep{episode + 1}.keras",
                q_agent_path=f"checkpoints/survival_q_agent_ep{episode + 1}.pkl"
            )

            # Calculate the completion rates
            avg_completion_100 = np.mean(completion_percentages[-100:])
            completion_rate_per_100.append(avg_completion_100)

            # Plot specifics
            plot_training_progress(
                all_rewards, 
                moving_avg_rewards, 
                all_survival_ratio,
                completion_percentages,
                completion_rate_per_100,
                episode + 1
            )

    # Save and plot the final iterations after completion of training
    agent.save_agents()
    
    plot_training_progress(
        all_rewards, 
        moving_avg_rewards, 
        all_survival_ratio,
        completion_percentages,
        completion_rate_per_100,
        episodes,
        final=True
    )
    
    # Print final training speed
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.0f} seconds")
    print(f"Total steps: {steps_taken}")
    print(f"Average speed: {steps_taken/total_time:.1f} steps/second")
    
    return agent


def plot_training_progress(rewards, moving_avg, survival_ratio, completion_rate_per_100, episode, final=False):
    """
    Function that plots all metrics relevant to the training of the agent.
    
    Args:
        rewards - rewards from the episodes
        moving_avg - the moving average per 100 episodes
        survival_ratio - % of how often the agent felt 'threatened'
        completion_rate_per_100 - % of level completion in 100 episodes span
        episode - current episode
        final - flag if this is the final step of training
    """
    plt.figure(figsize=(12, 10))
    
    # Episode reward plot given 100 episode moving average
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward', color='lightblue')
    plt.plot(moving_avg, linewidth=2, label='Moving Average (100 ep)', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress - Rewards')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # How much did the agent use the survival agent (1 - this % gives the exploration %)
    plt.subplot(2, 1, 2)
    plt.plot(survival_ratio, linewidth=2, label='Survival Mode Ratio', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Survival Mode Ratio')
    plt.title('Proportion of Actions Using Survival (PPO) Agent')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Show level completion rate
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(completion_rate_per_100)), completion_rate_per_100, 'mo-', linewidth=2)
    plt.xlabel('Checkpoint (x100 Episodes)')
    plt.ylabel('Avg Completion %')
    plt.title('Average Completion Rate Every 100 Episodes')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Agregate final scores after training
    if final:
        plt.savefig(f"survival_based_training_completion_rate_final.png")
    else:
        plt.savefig(f"checkpoints/survival_based_training_completion_rate_ep{episode}.png")

    plt.close()

    plt.tight_layout()
    
    # Save final plots
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