import tensorflow as tf
import numpy as np

def discount_rewards(rewards, gamma=0.99):
    """Discount future rewards."""
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    return discounted_rewards

def normalize(values):
    """Normalize a numpy array."""
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / (std + 1e-8)

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_epsilon=0.1, value_coeff=0.5, entropy_coeff=0.01, lam=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.lam = lam

        # Actor network (policy)
        self.actor = self._build_network(output_size=action_size, activation='softmax')
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Critic network (value function)
        self.critic = self._build_network(output_size=1, activation=None)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Experience buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_log_probs = []
        self.buffer_values = []

    def _build_network(self, output_size, activation, hidden_units=[64, 64]):
        layers = [tf.keras.layers.Input(shape=(self.state_size,))]
        for units in hidden_units:
            layers.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.BatchNormalization()
            ])
        layers.append(tf.keras.layers.Dense(output_size, activation=activation))
        return tf.keras.Sequential(layers)

    def choose_action(self, state):
        """Chooses an action based on the current state, considering valid actions."""
        # Get valid moves
        valid_moves_vector = state[-4:]
        valid_actions_indices = [i for i, val in enumerate(valid_moves_vector) if val == 1]
        
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

        # Predict policy
        policy = self.actor(state_tensor).numpy()[0]

        # Apply masking for valid actions
        mask = np.array([1 if i in valid_actions_indices else 0 for i in range(self.action_size)])
        policy *= mask
        # Renormalize the policy if some actions were masked
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        else:
            # If all actions are masked, choose a random valid action
            policy = np.zeros_like(policy)
            valid_indices = [i for i, val in enumerate(mask) if val == 1]
            if valid_indices:
                chosen_index = np.random.choice(valid_indices)
                policy[chosen_index] = 1.0
            else:
                print("Warning: No valid actions available.")
                return 0, -np.inf, 0.0  # Return a default action and log prob

        action = np.random.choice(self.action_size, p=policy)
        log_prob = np.log(policy[action] + 1e-8)
        value = self.critic(state_tensor)[0,0]

        return action, log_prob, value

    def learn(self, batch_size=64, epochs=10):
        """Updates the actor and critic networks based on the collected experiences."""
        # Convert buffer to numpy arrays
        states = np.array(self.buffer_states, dtype=np.float32)
        actions = np.array(self.buffer_actions, dtype=np.int32)
        rewards = np.array(self.buffer_rewards, dtype=np.float32)
        log_probs_old = np.array(self.buffer_log_probs, dtype=np.float32)  # Key fix
        values_old = np.array(self.buffer_values, dtype=np.float32).squeeze()
        dones = np.zeros_like(rewards)

        # Calculate GAE advantages and returns
        advantages = self.compute_gae(rewards, values_old, dones)
        returns = advantages + values_old
        advantages = normalize(advantages)

        for _ in range(epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_values_old = values_old[batch_indices]

                self._train_step(
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_log_probs_old,
                    batch_returns,
                    batch_values_old
                )

        # Clear buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_log_probs = []
        self.buffer_values = []

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        batch_size = len(rewards)
        advantages = np.zeros(batch_size, dtype=np.float32)
        last_advantage = 0

        # Handle bootstrapping
        next_value = 0 if dones[-1] else values[-1]

        for t in reversed(range(batch_size)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            next_value = values[t]

        return advantages

    @tf.function
    def _train_step(self, states, actions, advantages, log_probs_old, returns, values_old):
        with tf.GradientTape(persistent=True) as tape:
            # Actor loss
            policy = self.actor(states)
            casted_actions = tf.cast(actions, tf.int32)
            log_probs = tf.gather_nd(tf.math.log(policy + 1e-8), tf.stack([tf.range(tf.shape(casted_actions)[0]), casted_actions], axis=1))
            ratio = tf.exp(log_probs - log_probs_old)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Critic loss
            values = self.critic(states)[:, 0]
            critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

            # Entropy loss (encourage exploration)
            entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=1)
            entropy_loss = -tf.reduce_mean(entropy)

            total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def store_transition(self, state, action, reward, log_prob, value):
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    def save(self, actor_path="ppo_actor.h5", critic_path="ppo_critic.h5"):
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        print(f"PPO agent saved to {actor_path} and {critic_path}")

    def load(self, actor_path="ppo_actor.h5", critic_path="ppo_critic.h5"):
        try:
            self.actor = tf.keras.models.load_model(actor_path)
            self.critic = tf.keras.models.load_model(critic_path)
            print(f"PPO agent loaded from {actor_path} and {critic_path}")
        except Exception as e:
            print(f"Error loading PPO agent: {e}")