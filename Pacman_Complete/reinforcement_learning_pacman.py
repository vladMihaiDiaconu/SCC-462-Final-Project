import pygame
from pygame.locals import *
import numpy as np
import random
import time
import pickle

# Import your game components
from constants import *
from run import GameController  # Make sure this is the correct import for your game controller

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
        self.previous_pellets = 0
        self.previous_lives = self.game.lives
        self.pacman_was_alive = True
        
        # Start the game by unpausing
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
            return {
                'position': (0, 0),
                'direction': self.game.pacman.direction,
                'valid_moves': [],
                'ghosts': [],
                'score': self.game.score,
                'lives': self.game.lives
            }
        
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
        
        if action == UP:
            direction = UP
            key_event = K_UP
        elif action == DOWN:
            direction = DOWN
            key_event = K_DOWN
        elif action == LEFT:
            direction = LEFT
            key_event = K_LEFT
        elif action == RIGHT:
            direction = RIGHT
            key_event = K_RIGHT
        
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
        
        # Current game values
        current_score = self.game.score
        current_pellets = len(self.game.pellets.pelletList)
        current_lives = self.game.lives
        
        # Reward for score increase
        score_diff = current_score - self.previous_score
        if score_diff > 0:
            reward += score_diff * 0.1
        
        # Reward for eating pellets
        pellets_eaten = self.previous_pellets - current_pellets
        if pellets_eaten > 0:
            reward += pellets_eaten * 10
        
        # Penalty for losing lives
        lives_lost = self.previous_lives - current_lives
        if lives_lost > 0:
            reward -= 100
        
        # Reward for completing level
        if self.game.pellets.isEmpty():
            reward += 500
        
        # Small penalty to encourage faster completion
        reward -= 0.1
        
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


def train(episodes=100, max_steps=5000):
    """Train the agent on the Pac-Man game"""
    # Create environment
    env = PacmanWrapper()
    
    # Create agent
    agent = SimpleRLAgent()
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        done = False
        while not done and steps < max_steps:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done = env.take_action(action)
            
            # Learn from this experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Optional: slow down for visualization
            # time.sleep(0.05)
        
        # End of episode
        print(f"Episode {episode + 1}/{episodes}, Score: {state['score']}, Reward: {total_reward:.2f}")
        
        # Save periodically
        if (episode + 1) % 25 == 0:
            agent.save(f"pacman_agent_ep{episode + 1}.pkl")
    
    # Final save
    agent.save("pacman_agent_final.pkl")
    
    return agent


if __name__ == "__main__":
    # Train the agent
    train(episodes=100)