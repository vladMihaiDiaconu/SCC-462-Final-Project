import numpy as np
import random

# Define GridWorld parameters
GRID_SIZE = 5
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)]  # Right, Left, Down, Up

# ------------------------ Environment Setup ------------------------
def initialize_environment():
    """
    Initializes the Pac-Man environment with:
    - Pac-Man at (0,0)
    - Randomly placed ghosts
    - Randomly placed pellets
    """
    agent_pos = (0, 0)  # Pac-Man starts at top-left corner
    ghosts = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(2)]
    pellets = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(3)]
    
    # Ensure Pac-Man doesn't start on a ghost or pellet
    if agent_pos in ghosts:
        ghosts.remove(agent_pos)
    if agent_pos in pellets:
        pellets.remove(agent_pos)
    
    return agent_pos, ghosts, pellets

def distance(pos1, pos2):
    """Computes Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# ------------------------ Threat Index Calculation ------------------------
def compute_threat_index(agent_pos, ghosts, max_threat=1.5):
    """
    Computes the Threat Index based on distance to ghosts.
    - Closer ghosts increase the threat level.
    - The index is normalized to [0,1].
    """
    threat_sum = 0
    epsilon = 1e-5  # Avoid division by zero

    for ghost in ghosts:
        threat_sum += 1 / (distance(agent_pos, ghost) + epsilon)  # Closer ghost = higher threat
    
    return min(threat_sum / max_threat, 1.0)  # Normalize to [0,1]

# ------------------------ RL Policies (Q-Learning & PPO) ------------------------
def q_learning_policy(state):
    """Q-learning policy: Chooses a random action (exploration)."""
    return random.choice(ACTIONS)  # Random move (exploration)

def ppo_policy(state, agent_pos, ghosts):
    """PPO policy: Avoids ghosts by moving away from them."""
    max_dist = 0
    best_action = None

    for action in ACTIONS:
        new_pos = (max(0, min(agent_pos[0] + action[0], GRID_SIZE-1)),
                   max(0, min(agent_pos[1] + action[1], GRID_SIZE-1)))
        min_dist_to_ghost = min([distance(new_pos, ghost) for ghost in ghosts])
        
        if min_dist_to_ghost > max_dist:
            max_dist = min_dist_to_ghost
            best_action = action

    return best_action if best_action else random.choice(ACTIONS)  # Default to random if uncertain

# ------------------------ Hard Switching Mechanism ------------------------
def switch_policies(state, threat_index, agent_pos, ghosts, threshold=0.5):
    """
    Hard switches between Q-learning and PPO based on Threat Index.
    
    threshold: T(s) above this means PPO takes over.
    """
    if threat_index < threshold:
        return q_learning_policy(state)  # Exploration mode
    else:
        return ppo_policy(state, agent_pos, ghosts)  # Survival mode

# ------------------------ Running the Experiment ------------------------
EPISODES = 50  # Number of episodes
results = {"survival_rate": 0, "pellets_collected": 0, "avg_episode_length": 0}

for episode in range(EPISODES):
    agent_pos, ghosts, pellets = initialize_environment()
    steps = 0
    survived = True
    pellets_collected = 0

    while steps < 50:  # Max steps per episode
        threat_index = compute_threat_index(agent_pos, ghosts)  # Compute threat level
        action = switch_policies(agent_pos, threat_index, agent_pos, ghosts)  # Select action
        
        # Move agent
        new_pos = (max(0, min(agent_pos[0] + action[0], GRID_SIZE-1)), 
                   max(0, min(agent_pos[1] + action[1], GRID_SIZE-1)))

        # Check if agent hits a ghost (death)
        if new_pos in ghosts:
            survived = False
            break  # Episode ends

        # Check if agent collects a pellet
        if new_pos in pellets:
            pellets_collected += 1
            pellets.remove(new_pos)  # Remove collected pellet

        agent_pos = new_pos
        steps += 1

    # Store results
    results["survival_rate"] += 1 if survived else 0
    results["pellets_collected"] += pellets_collected
    results["avg_episode_length"] += steps

# Compute averages
results["survival_rate"] /= EPISODES
results["pellets_collected"] /= EPISODES
results["avg_episode_length"] /= EPISODES

# Display results
print("\n==== EXPERIMENT RESULTS ====")
print(f"Survival Rate: {results['survival_rate']:.2f}")
print(f"Avg Pellets Collected: {results['pellets_collected']:.2f}")
print(f"Avg Episode Length: {results['avg_episode_length']:.2f}")
