import pygame
from pygame.locals import *
import random
import time
import pickle
import math
import collections
import os

# Import the game components
from constants import *
from run import GameController

class PacmanWrapper:
    """
        A simple wrapper for the Pac-Man game that handles game control
        and provides a clean interface for reinforcement learning.
        """
    ###- New To the Original Qlearninr_basic code
    LOOP_WINDOW = 8   # timesteps to look back when detecting ping-pong
    LOOP_PENALTY = 1.0  # penalty applied when Pac‑Man bounces back and forth
    ###-

    def __init__(self):
        # Initialize game
        self.game = GameController()
        self.game.startGame()

        # Initialize state variables
        self.previous_score   = 0
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives   = self.game.lives
        self.pacman_was_alive = True

        ###- New To the Original Qlearning_basic code
        # Exploration bookkeeping
        self.visit_counts     = collections.Counter()               # count‑based bonus
        self.recent_positions = collections.deque(maxlen=self.LOOP_WINDOW)
        self.prev_direction   = STOP                                # for reverse & corridor checks
        ###-

        # Start the game by unpausing
        self.unpause_game()

    # Unpause helper
    def unpause_game(self):

        # Press Space and Enter key multiple times to make sure game starts
        for _ in range(5):
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
            pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
            time.sleep(0.1)
            self.game.update()

            # If game is unpaused, we're good to go
            if not self.game.pause.paused:
                break

    # Death handler
    def check_if_pacman_died(self):
        """Check if Pac-Man just died and handle it"""
        # Detect if Pac-Man just died
        if self.pacman_was_alive and not self.game.pacman.alive:

            # Wait a bit for death animation
            time.sleep(0.5)
            # Try to unpause repeatedly after death
            for _ in range(10):
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_SPACE}))
                pygame.event.post(pygame.event.Event(KEYDOWN, {'key': K_RETURN}))
                time.sleep(0.1)
                self.game.update()
                # If game unpaused, we can stop
                if not self.game.pause.paused:
                    break
        # Update alive status for next check
        self.pacman_was_alive = self.game.pacman.alive


    # get_state
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
        pacman = self.game.pacman  #pacman variable creation for better hadling
        if pacman.node is None:
            # Minimal state if Pac‑Man is not on a node
            return {
                'position': (0, 0),
                'direction': pacman.direction,
                'valid_moves': [],
                'ghosts': [],
                'score': self.game.score,
                'lives': self.game.lives,
                ###- New To the Original Qlearninr_basic code
                'pellet_here': False
                ###-
            }

        # Get Pac-Man's position and valid moves
        position = (int(pacman.position.x), int(pacman.position.y))
        valid_moves = pacman.validDirections()

        ghosts = []
        for ghost in self.game.ghosts:
            ghost_info = {
                'position': (int(ghost.position.x), int(ghost.position.y)),
                'frightened': ghost.mode.current is FREIGHT
            }
            ghosts.append(ghost_info)

        pellet_here = False
        for p in self.game.pellets.pelletList:
            if int(p.position.x) == position[0] and int(p.position.y) == position[1]:
                pellet_here = True
                break

        # Return the state
        return {
            'position': position,
            'direction': pacman.direction,
            'valid_moves': valid_moves,
            'ghosts': ghosts,
            'score': self.game.score,
            'lives': self.game.lives,
            'pellet_here': pellet_here
        }

    # take_action fucntion
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

        pacman = self.game.pacman
        # Allow movement only when Pac‑Man is at a node and idle
        if pacman.node is not None and pacman.target is None:
            next_node = pacman.node.neighbors.get(action)
            if next_node:
                pacman.direction = action
                pacman.target = next_node

        # Update game step
        self.game.update()
        # finalize move when target reached
        if pacman.target and pacman.position == pacman.target.position:
            pacman.node = pacman.target
            pacman.target = None

        # At this point we can debug to see what is happening
        # print(f"Pacman: pos={pacman.position}, node={pacman.node.position if pacman.node else None}, target={pacman.target}")
        # Get reward
        reward = self.calculate_reward(action)

        # Get next state
        next_state = self.get_state()

        # Check if episode is done
        done = self.game.lives <= 0 or self.game.pellets.isEmpty()

        return next_state, reward, done


    # calculate_reward : pellets boosts, corridors bonus, novelty in the taken path
    def calculate_reward(self, action):
        """Reward based on game events and exploration bonuses
        To encourage Dora to explore more"""
        # Current game values
        current_score   = self.game.score
        current_pellets = len(self.game.pellets.pelletList)
        current_lives   = self.game.lives

        ###- New To the Original Qlearninr_basic code) this in is for better handling
        pac_pos = (int(self.game.pacman.position.x), int(self.game.pacman.position.y))
        ###-

        reward = 0
        # Reward for score increase
        reward += 0.01 * (current_score - self.previous_score)
        # Reward for eating pellets (gives to Dora +12)
        pellets_eaten = self.previous_pellets - current_pellets
        reward += 1.2 * pellets_eaten

        # Reward for corridors (gives to Dora +2)
        if pellets_eaten > 0 and action == self.prev_direction and action != STOP:
            reward += 0.2

        # Penalty for losing lives (penalizes to Dora -100) optimization in variable

        reward -= 5 * (self.previous_lives - current_lives)

        # Reward for completing level
        if self.game.pellets.isEmpty():
            reward += 5.0

        # Small penalty to encourage faster completion
        reward -= 0.05

        ###- New To the Original Qlearninr_basic code

        # Intrinsic exploration bonus – only if pellet still here
        self.visit_counts[pac_pos] += 1
        if any(int(p.position.x)==pac_pos[0] and int(p.position.y)==pac_pos[1]
               for p in self.game.pellets.pelletList):
            reward += 1 / self.visit_counts[pac_pos]

        # Loop deterrent
        if len(self.recent_positions)>=self.LOOP_WINDOW and len(set(self.recent_positions))<=2:
            reward -= self.LOOP_PENALTY
        self.recent_positions.append(pac_pos)
        # Reverse‑move penalty
        if action == {UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT}.get(self.prev_direction, STOP):
            reward -= 0.2
        self.prev_direction = action

        ###-

        # Update previous values
        self.previous_score   = current_score
        self.previous_pellets = current_pellets
        self.previous_lives   = current_lives

        return reward

    # reset
    def reset(self):
        """Reset the game for a new episode"""
        # Reset game if needed
        if self.game.lives <= 0:
            self.game.restartGame()

        # Make sure game is unpaused
        self.unpause_game()

        # Reset state variables
        pacman = self.game.pacman
        pacman.setPosition()
        pacman.direction = STOP
        pacman.target = None

        self.previous_score   = self.game.score
        self.previous_pellets = len(self.game.pellets.pelletList)
        self.previous_lives   = self.game.lives
        self.pacman_was_alive = True

        ###- New To the Original Qlearninr_basic code
        # Clear Variables (To be sure)
        self.visit_counts.clear(); self.recent_positions.clear(); self.prev_direction = STOP
        ###-

        # Return initial state
        return self.get_state()

# SimpleRLAgent function to apply : epsilon-greedy + UCB

class SimpleRLAgent:
    """
        A very simple RL agent for Pac‑Man using Q‑learning + UCB tie‑breaking
        In classic Epsilon Greedy, ties between Qvalues are broken randomly, or picked arbitrarily.
        But if multiple actions have similar Qvalues, we want to pick the least explored one, and we will do
        so using UCB
    """
    def __init__(self):
        # Will store Q-values for state-action pairs
        self.q_table = {}
        # Learning parameters
        self.learning_rate   = 0.15 #before 0.1
        self.discount_factor = 0.99 #before 0.95
        # Exploration rate
        self.epsilon       = 1.0   # Exploration rate / Dora'll start fully exploratory
        self.epsilon_decay = 0.9999 # To keep Dora as explorer as possible in most episodes
        self.epsilon_min   = 0.4
        ###- New To the Original Qlearninr_basic code
        # UCB bookkeeping
        self.state_counts = collections.Counter()
        self.state_action_counts = collections.Counter()
        self.ucb_c = 1.0  # exploration strength for UCB
        self.steps   = 0
        ###-

    # Helper to convert state dict to hashable key
    def state_to_key(self, state):
        """Convert state dict to a hashable key for the Q-table"""
        # Create a simplified state representation for the Q-table
        # Focus on Pac-Man's position and ghost positions
        pacman_pos = (state['position'][0] // 16, state['position'][1] // 16)

        # Simplifyded ghost representation
        ghosts = tuple(((g['position'][0] // 16, g['position'][1] // 16), g['frightened']) for g in state['ghosts'])

        # Create a hashable key
        return (pacman_pos, ghosts, state['lives'], state['pellet_here'])

    # choose an action using Epsilon greedy with UCB when exploiting
    def choose_action(self, state):
        # Get valid moves
        valid_moves = state['valid_moves']

        # If no valid moves, return current direction to keep moving
        if not valid_moves:
            return state['direction']

        ###- New To the Original Qlearninr_basic code

        # Exploit: choose best action based on Q-values
        key = self.state_to_key(state)
        self.state_counts[key] += 1
        self.steps += 1

        # Initialize Q-values if state is new
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in [UP, DOWN, LEFT, RIGHT]}

        # Epsilon greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose random action
            action = random.choice(valid_moves)
            #keep track of actions
            self.state_action_counts[(key, action)] += 1
            return action

        # UCB tie‑breaker among best Q actions
        total_visits = self.state_counts[key] # Number of times the current state (key) has been visited
        best_action, best_ucb = None, -float('inf') # Initializes the variables for best action and the highest UCB score
        # Here we loop through all valid moves and track the one with the maximum UCB value
        for a in valid_moves:
            q = self.q_table[key][a]  # current Qvalue estimate
            n_sa = self.state_action_counts[(key, a)]  # looks up how often action a was taken in state key
            ucb_val = q + self.ucb_c * math.sqrt(math.log(total_visits + 1) / (1 + n_sa)) #UBC formula
            if ucb_val > best_ucb: # to checks if the current action has the highest UCB score so far
                best_ucb, best_action = ucb_val, a # updates the current best action and its UCB score
        # Now we increment the count of how often this action was selected in this state
        self.state_action_counts[(key, best_action)] += 1

        return best_action

        ###-

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        # Convert states to hashable keys
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        # Initialize Q-values if states are new
        for k in (state_key, next_state_key):
            if k not in self.q_table:
                self.q_table[k] = {a: 0.0 for a in [UP, DOWN, LEFT, RIGHT]}

        # Get actual Qvalues and max the next Qvalue
        q_sa = self.q_table[state_key][action]
        max_q_next = max(self.q_table[next_state_key].values())

        #Calculate the target Qvalue
        if done: #the targer will be the reward since there is no future reward
            target = reward
        else: #Use bellman equation
            target = reward + self.discount_factor * max_q_next

        # Update the Qvalue torward the target
        self.q_table[state_key][action] += self.learning_rate * (target - q_sa)

        #Decay exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon - 0.001)

    # Save and Load Files
    def save(self, filename="Dora_Agent.pkl"):
        """Save the agent to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        print(f"Agent saved to {filename}")

    def load(self, filename="Dora_Agent.pkl"):
        """Load the agent from a file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
        except Exception as e:
            print(f"Error loading agent: {e}")


def train(episodes=1000, max_steps=3000):
    """Train the agent on the Pac-Man game"""
    # Create environment
    env   = PacmanWrapper()

    # Create agent
    agent = SimpleRLAgent()

    # As backup (checkpoints created)
    os.makedirs('checkpoints', exist_ok=True)

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
            state, total_reward, steps = next_state, total_reward + reward, steps + 1

        # End of episode summary
        print(f"Episode {episode + 1}/{episodes} | Score: {state['score']} | Total Reward: {total_reward:.2f}")

        # Save periodically
        if (episode + 1) % 25 == 0:
            agent.save(f"checkpoints/pacman_agent_ep{episode + 1}.pkl")

    # Final save
    agent.save("Dora_Agent.pkl")


if __name__ == "__main__":
    # Train the agent
    train(episodes=600)
