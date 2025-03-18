import numpy as np
import random

# Define GridWorld parameters
GRID_SIZE = 5
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)]  # Right, Left, Down, Up
ALPHA = 0.1  # Learning rate for threshold adjustment
BETA = 0.05  # Randomness factor for non-determinism

# ------------------------ Environment Setup ------------------------
def initialize_environment():
    """Initializes the environment."""
    agent_pos = (0, 0)
    ghosts = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(2)]
    pellets = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(3)]
    
    if agent_pos in ghosts:
        ghosts.remove(agent_pos)
    if agent_pos in pellets:
        pellets.remove(agent_pos)
    
    return agent_pos, ghosts, pellets

def distance(pos1, pos2):
    """Computes Euclidean distance."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# ------------------------ Threat Index Calculation ------------------------
def compute_threat_index(agent_pos, ghosts, max_threat=1.5):
    """Computes Threat Index based on distance to threats."""
    threat_sum = 0
    epsilon = 1e-5  

    for ghost in ghosts:
        threat_sum += 1 / (distance(agent_pos, ghost) + epsilon)  
    
    return min(threat_sum / max_threat, 1.0)  

# ------------------------ Non-Deterministic Adaptive Thresholding ------------------------
def update_threshold(threshold, survival_rate, pellet_collection_rate, alpha=ALPHA, beta=BETA):
    """
    Updates the threshold dynamically with stochastic adjustments.
    - Uses a weighted sum of survival & exploration performance.
    - Adds a small random variation to avoid determinism.
    """
    change = alpha * (survival_rate - pellet_collection_rate) + beta * np.random.normal(0, 1)
    new_threshold = np.clip(threshold + change, 0.2, 0.8)  # Keep within reasonable bounds
    return new_threshold

# ------------------------ RL Policies ------------------------
def q_learning_policy(state):
    """Q-learning policy: Chooses a random action (exploration)."""
    return random.choice(ACTIONS)  

def ppo_policy(state, agent_pos, ghosts):
    """PPO policy: Moves away from the closest ghost."""
    max_dist = 0
    best_action = None

    for action in ACTIONS:
        new_pos = (max(0, min(agent_pos[0] + action[0], GRID_SIZE-1)),
                   max(0, min(agent_pos[1] + action[1], GRID_SIZE-1)))
        min_dist_to_ghost = min([distance(new_pos, ghost) for ghost in ghosts])
        
        if min_dist_to_ghost > max_dist:
            max_dist = min_dist_to_ghost
            best_action = action

    return best_action if best_action else random.choice(ACTIONS)  

def evasive_maneuver(state, agent_pos, ghosts):
    """Pre-learned zig-zag escape maneuver."""
    return random.choice([(1,1), (-1,-1), (1,-1), (-1,1)])  

def random_escape(state):
    """Randomized erratic movement to confuse threats."""
    return random.choice(ACTIONS)  

# ------------------------ Multi-Modal Survival with Adaptive Threshold ------------------------
def multi_modal_survival(state, threat_index, agent_pos, ghosts, threshold):
    """
    Multi-modal survival switching:
    - If safe → Exploration
    - If escape possible → PPO
    - If trapped → Evasive maneuver
    - If ghost is unpredictable → Random escape
    """
    if threat_index < threshold:
        return q_learning_policy(state)  
    else:
        escape_possible = any(
            (max(0, min(agent_pos[0] + action[0], GRID_SIZE-1)),
             max(0, min(agent_pos[1] + action[1], GRID_SIZE-1))) not in ghosts
            for action in ACTIONS
        )
        
        if escape_possible:
            return ppo_policy(state, agent_pos, ghosts)  
        elif random.random() < 0.5:  
            return evasive_maneuver(state, agent_pos, ghosts)  
        else:
            return random_escape(state)  

# ------------------------ Running the Experiment ------------------------
EPISODES = 50  
threshold = 0.5  

results = {"survival_rate": 0, "pellets_collected": 0, "avg_episode_length": 0}

for episode in range(EPISODES):
    agent_pos, ghosts, pellets = initialize_environment()
    steps = 0
    survived = True
    pellets_collected = 0

    while steps < 50:  
        threat_index = compute_threat_index(agent_pos, ghosts)  
        action = multi_modal_survival(agent_pos, threat_index, agent_pos, ghosts, threshold)  

        new_pos = (max(0, min(agent_pos[0] + action[0], GRID_SIZE-1)), 
                   max(0, min(agent_pos[1] + action[1], GRID_SIZE-1)))

        if new_pos in ghosts:
            survived = False
            break  

        if new_pos in pellets:
            pellets_collected += 1
            pellets.remove(new_pos)  

        agent_pos = new_pos
        steps += 1

    # Update results
    results["survival_rate"] += 1 if survived else 0
    results["pellets_collected"] += pellets_collected
    results["avg_episode_length"] += steps

    # Update threshold dynamically
    survival_rate = results["survival_rate"] / (episode + 1)
    pellet_collection_rate = results["pellets_collected"] / (episode + 1)
    threshold = update_threshold(threshold, survival_rate, pellet_collection_rate)

print(f"Final Threshold: {threshold:.2f}")
print(f"Survival Rate: {results['survival_rate'] / EPISODES:.2f}")
print(f"Avg Pellets Collected: {results['pellets_collected'] / EPISODES:.2f}")
