import pygame
from pygame.locals import *
import numpy as np
import random
import time
import pickle

# Import the game components
from constants import *
from run import GameController 
from ppo import PPOAgent
from collections import deque
import matplotlib.pyplot as plt

class PacmanWrapper:
    """
    A simple wrapper for the Pac-Man game that handles game control
    and provides a clean interface for reinforcement learning.
    """
    def __init__(self, agent_type="survival"):
        # Initialize game
        self.game = GameController()
        self.game.startGame()
        
        # Initialize state variables
        self.previous_score = 0
        self.previous_pellets = 0
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        self.agent_type = agent_type

        self.last_positions = deque(maxlen=4)  
        self.last_direction = STOP
        self.unpause_game()
        
    def unpause_game(self):
        """Make sure the game is unpaused and ready to play"""
        print("Trying to unpause game...")
        
        # Press Space key multiple times to make sure game starts
        for _ in range(5):
            # Create and post a Space key event
            event = pygame.event.Event(KEYDOWN, {'key': K_SPACE})
            pygame.event.post(event)
            time.sleep(0.1)
            self.game.update()
            
            # Also try Enter key
            event = pygame.event.Event(KEYDOWN, {'key': K_RETURN})
            pygame.event.post(event)
            time.sleep(0.1)
            self.game.update()
            
            # If game is unpaused, we're good to go
            if not self.game.pause.paused:
                print("Game unpaused successfully")
                break
    
    def check_if_pacman_died(self):
        """Check if Pac-Man just died and handle it"""
        # Detect if Pac-Man just died
        if self.pacman_was_alive and not self.game.pacman.alive:
            print("Pac-Man died! Handling death...")
            
            # Wait a bit for death animation
            time.sleep(0.5)
            
            # Try to unpause repeatedly after death
            for _ in range(10):
                # Space key is crucial after death
                event = pygame.event.Event(KEYDOWN, {'key': K_SPACE})
                pygame.event.post(event)
                time.sleep(0.2)
                self.game.update()
                
                # Also try Enter
                event = pygame.event.Event(KEYDOWN, {'key': K_RETURN})
                pygame.event.post(event)
                time.sleep(0.1)
                self.game.update()
                
                # If game unpaused, we can stop
                if not self.game.pause.paused:
                    print("Game unpaused after death")
                    break
        
        # Update alive status for next check
        self.pacman_was_alive = self.game.pacman.alive
    
    def get_state(self):
        """
        Get current game state in a format useful for RL
        Returns a simplified representation of the game state
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()
        
        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()
        
        # If Pac-Man doesn't have a valid node, return a minimal state
        if self.game.pacman.node is None:
            return np.zeros(20) if self.agent_type == "survival" else {
                'position': (0, 0),
                'direction': self.game.pacman.direction,
                'valid_moves': [],
                'ghosts': [],
                'score': self.game.score,
                'lives': self.game.lives
            }

        if self.agent_type == "survival":
            pacman_x = self.game.pacman.position.x / SCREENWIDTH
            pacman_y = self.game.pacman.position.y / SCREENHEIGHT 
            direction = np.zeros(4)
            if self.game.pacman.direction == UP:
                direction[0] = 1
            elif self.game.pacman.direction == DOWN:
                direction[1] = 1
            elif self.game.pacman.direction == LEFT:
                direction[2] = 1
            elif self.game.pacman.direction == RIGHT:
                direction[3] = 1
            ghost_features = []
            for ghost in self.game.ghosts:
                rel_x = (ghost.position.x - self.game.pacman.position.x) / SCREENWIDTH
                rel_y = (ghost.position.y - self.game.pacman.position.y) / SCREENHEIGHT 
                frightened = 1 if ghost.mode.current is FREIGHT else 0
                ghost_features.extend([rel_x, rel_y, frightened])
            while len(ghost_features) < 12:
                ghost_features.append(0)
            pellet_distances = []
            if self.game.pacman.node:
                for pellet in self.game.pellets.pelletList:
                    dist = (self.game.pacman.position - pellet.position).magnitude()
                    pellet_distances.append(dist)
                pellet_distances.sort()
                closest_pellets = pellet_distances[:3]
                pellet_features = [(d / SCREENWIDTH) for d in closest_pellets]
                while len(pellet_features) < 3:
                    pellet_features.append(1.0)
            else:
                pellet_features = [1.0] * 3
            
            # Get valid moves
            valid_moves_list = []
            for move in [UP, DOWN, LEFT, RIGHT]:
                if self.game.pacman.node.neighbors[move] is not None:
                    valid_moves_list.append(move)

            # Convert valid moves to a one-hot encoded vector
            valid_moves_vector = np.zeros(4, dtype=np.float32)
            for move in valid_moves_list:
                if move == UP:
                    valid_moves_vector[0] = 1
                elif move == DOWN:
                    valid_moves_vector[1] = 1
                elif move == LEFT:
                    valid_moves_vector[2] = 1
                elif move == RIGHT:
                    valid_moves_vector[3] = 1

            state = np.concatenate([
                [pacman_x, pacman_y],
                direction,
                ghost_features,
                pellet_features,
                [self.game.lives / 3.0],
                valid_moves_vector
            ], dtype=np.float32)
            return state
        else:  # agent_type == "exploration"
            # Get Pac-Man's position and valid moves
            position = (
                int(self.game.pacman.position.x),
                int(self.game.pacman.position.y)
            )

            # Get valid moves
            valid_moves = []
            for direction in [UP, DOWN, LEFT, RIGHT]:
                if self.game.pacman.node.neighbors[direction] is not None:
                    valid_moves.append(direction)
            
            # Get ghost information
            ghosts = []
            for ghost in self.game.ghosts:
                ghost_info = {
                    'position': (int(ghost.position.x), int(ghost.position.y)),
                    'frightened': ghost.mode.current is FREIGHT
                }
                ghosts.append(ghost_info)
            
            # Return the state
            return {
                'position': position,
                'direction': self.game.pacman.direction,
                'valid_moves': valid_moves,
                'ghosts': ghosts,
                'score': self.game.score,
                'lives': self.game.lives
            }
    
    def take_action(self, action):
        """
        Execute an action in the game
        action should be one of: UP, DOWN, LEFT, RIGHT (0, 1, 2, 3)
        """
        # Handle Pac-Man death first
        self.check_if_pacman_died()
        
        # If game is paused, try to unpause it
        if self.game.pause.paused:
            self.unpause_game()
        
        # Map action number to direction
        direction = None
        key_event = None
        
        if action == 0:
            direction = UP
            key_event = K_UP
        elif action == 1:
            direction = DOWN
            key_event = K_DOWN
        elif action == 2:
            direction = LEFT
            key_event = K_LEFT
        elif action == 3:
            direction = RIGHT
            key_event = K_RIGHT
        elif self.agent_type == "exploration":
            if action == UP: direction = UP; key_event = K_UP
            elif action == DOWN: direction = DOWN; key_event = K_DOWN
            elif action == LEFT: direction = LEFT; key_event = K_LEFT
            elif action == RIGHT: direction = RIGHT; key_event = K_RIGHT
        
        # Only change direction if Pac-Man has a valid node and the move is valid
        if direction is not None and self.game.pacman.node is not None:
            if self.game.pacman.node.neighbors[direction] is not None:
                # Send key event
                event = pygame.event.Event(KEYDOWN, {'key': key_event})
                pygame.event.post(event)
                
                # Set direction directly
                self.game.pacman.direction = direction
        
        # Update the game
        self.game.update()
        
        # Get reward
        reward = self.calculate_reward()
        
        # Get next state
        next_state = self.get_state()
        
        # Check if episode is done
        done = self.game.lives <= 0 or self.game.pellets.isEmpty()
        
        return next_state, reward, done
    
    def calculate_reward(self):
        """Calculate reward based on game events"""
        reward = 0
        pacman = self.game.pacman
        ghosts = self.game.ghosts
        pellets_remaining = len(self.game.pellets.pelletList)

        score_diff = self.game.score - self.previous_score
        lives_lost = self.previous_lives - self.game.lives
        level_completed = pellets_remaining == 0

        if self.agent_type == "survival":
            # Reward for increasing score (includes eating pellets and ghosts)
            reward += score_diff * 0.1

            # Penalty for losing lives
            if lives_lost > 0:
                reward -= 100

            # Reward for completing the level
            if level_completed:
                reward += 500

            # Small reward for surviving each step
            reward += 0.01

            # Penalty for being very close to a non-frightened ghost
            min_ghost_distance = float('inf')
            for ghost in ghosts:
                if ghost.mode.current != FREIGHT:
                    dist = (ghost.position - pacman.position).magnitude()
                    min_ghost_distance = min(min_ghost_distance, dist)

            if min_ghost_distance < TILEWIDTH * 1.5:
                reward -= 0.2

        elif self.agent_type == "exploration":
            # Reward for increasing score
            reward += score_diff * 0.05

            # Small negative reward per step to encourage movement
            reward -= 0.01

            # Reward for eating pellets (contributes to exploration)
            if self.previous_pellets > pellets_remaining:
                reward += (self.previous_pellets - pellets_remaining) * 5

            # Reward for completing the level
            if level_completed:
                reward += 200

        # Update previous values
        self.previous_score = self.game.score
        self.previous_lives = self.game.lives
        self.previous_pellets = pellets_remaining

        return reward
    
    def reset(self):
        """Reset the game for a new episode"""
        # Reset game if needed
        if self.game.lives <= 0:
            self.game.restartGame()
        
        # Make sure game is unpaused
        self.unpause_game()
        
        # Reset state variables
        self.previous_score = self.game.score
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        
        # Return initial state
        return self.get_state()


class SimpleRLAgent:
    """
    A very simple RL agent for Pac-Man using Q-learning
    This is just a template that you can replace with your own algorithm
    """
    def __init__(self):
        self.q_table = {}  # Will store Q-values for state-action pairs
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        self.last_action = None
    
    def state_to_key(self, state):
        """Convert state dict to a hashable key for the Q-table"""
        # Create a simplified state representation for the Q-table
        # Focus on Pac-Man's position and ghost positions
        pacman_pos = state['position']
        
        # Simplify ghost representation
        ghosts = tuple(
            (g['position'], g['frightened'])
            for g in state['ghosts']
        )
        
        # Create a hashable key
        return (pacman_pos, ghosts, state['lives'])
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        # Get valid moves
        valid_moves = state['valid_moves']
        
        # If no valid moves, return current direction
        if not valid_moves:
            return state['direction']
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose random action
            action = random.choice(valid_moves)
        else:
            # Exploit: choose best action based on Q-values
            state_key = self.state_to_key(state)
            
            # Initialize Q-values if state is new
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in [UP, DOWN, LEFT, RIGHT]}
            
            # Find best action among valid moves
            q_values = self.q_table[state_key]
            best_action = max(valid_moves, key=lambda a: q_values.get(a, 0))
            action = best_action
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Remember this action
        self.last_action = action
        
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        # Convert states to hashable keys
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Initialize Q-values if states are new
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in [UP, DOWN, LEFT, RIGHT]}
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in [UP, DOWN, LEFT, RIGHT]}
        
        # Q-learning update rule
        if not done:
            # Get best action value from next state
            best_next_action_value = max(self.q_table[next_state_key].values())
            
            # Update Q-value
            self.q_table[state_key][action] += self.learning_rate * (
                reward + self.discount_factor * best_next_action_value - self.q_table[state_key][action]
            )
        else:
            # Terminal state update
            self.q_table[state_key][action] += self.learning_rate * (
                reward - self.q_table[state_key][action]
            )
    
    def save(self, filename="pacman_agent.pkl"):
        """Save the agent to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        print(f"Agent saved to {filename}")
    
    def load(self, filename="pacman_agent.pkl"):
        """Load the agent from a file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
            print(f"Agent loaded from {filename}")
        except Exception as e:
            print(f"Error loading agent: {e}")


# def train(episodes=100, max_steps=5000):
#     """Train the agent on the Pac-Man game"""
#     # Create environment
#     env = PacmanWrapper()
    
#     # Create agent
#     agent = SimpleRLAgent()
    
#     # Training loop
#     for episode in range(episodes):
#         # Reset environment
#         state = env.reset()
#         total_reward = 0
#         steps = 0
        
#         # Episode loop
#         done = False
#         while not done and steps < max_steps:
#             # Choose action
#             action = agent.choose_action(state)
            
#             # Take action
#             next_state, reward, done = env.take_action(action)
            
#             # Learn from this experience
#             agent.learn(state, action, reward, next_state, done)
            
#             # Update state and counters
#             state = next_state
#             total_reward += reward
#             steps += 1
            
#             # Optional: slow down for visualization
#             # time.sleep(0.05)
        
#         # End of episode
#         print(f"Episode {episode + 1}/{episodes}, Score: {state['score']}, Reward: {total_reward:.2f}")
        
#         # Save periodically
#         if (episode + 1) % 25 == 0:
#             agent.save(f"pacman_agent_ep{episode + 1}.pkl")
    
#     # Final save
#     agent.save("pacman_agent_final.pkl")
    
#     return agent


# if __name__ == "__main__":
#     # Train the agent
#     train(episodes=100)


def train(agent, env, total_episodes=1000, max_steps=2000, batch_size=256, save_interval=50):
    """Train the agent on the Pac-Man game"""
    # Training metrics
    episode_rewards = []
    moving_avg = []
    best_avg_reward = -np.inf
    global_step = 0

    # Warm-up run (optional)
    print("Running warm-up...")
    _ = env.reset()
    for _ in range(100):
        action = random.randint(0, 3)
        _, _, _ = env.take_action(action)

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_steps:
            # Get action from agent
            if isinstance(agent, PPOAgent):
                action, log_prob, value = agent.choose_action(state)
            else:
                action = agent.choose_action(state)

            # Environment step
            next_state, reward, done = env.take_action(action)
            episode_reward += reward
            episode_steps += 1
            global_step += 1

            # Store experience
            if isinstance(agent, PPOAgent):
                agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob
                )
                
                # Learn if batch is ready
                if len(agent.buffer_states) >= batch_size:
                    agent.learn()
            
            elif isinstance(agent, SimpleRLAgent):
                agent.learn(state, action, reward, next_state, done)

            state = next_state

        # Episode complete
        episode_rewards.append(episode_reward)
        current_avg = np.mean(episode_rewards[-100:])
        moving_avg.append(current_avg)

        # Save best model
        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            if isinstance(agent, PPOAgent):
                agent.save("best_actor.h5", "best_critic.h5")

        # Progress reporting
        if episode % save_interval == 0 or episode == 1:
            print(f"Ep {episode} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg 100: {current_avg:.1f} | "
                  f"Steps: {global_step}")
            
            if isinstance(agent, PPOAgent):
                agent.save(f"checkpoint_{episode}_actor.h5", 
                          f"checkpoint_{episode}_critic.h5")

    # Final save and plot
    if isinstance(agent, PPOAgent):
        agent.save("final_actor.h5", "final_critic.h5")
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(moving_avg, linewidth=2, label='100-Episode Avg')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("training_progress.png")
    plt.close()

    return agent

if __name__ == "__main__":
    episodes_survival = 50
    episodes_exploration = 50
    max_steps = 5000

    # Train Survival Agent (PPO)
    env_survival = PacmanWrapper(agent_type="survival")
    state_size_survival = env_survival.get_state().shape[0]
    action_size = 4
    survival_agent = PPOAgent(state_size_survival, action_size)
    trained_survival_agent = train(survival_agent, env_survival, episodes_survival, max_steps)

    # Train Exploration Agent (SimpleRL)
    env_exploration = PacmanWrapper(agent_type="exploration")
    exploration_agent = SimpleRLAgent()
    trained_exploration_agent = train(exploration_agent, env_exploration, episodes_exploration, max_steps)

    print("\n--- Training Complete ---")