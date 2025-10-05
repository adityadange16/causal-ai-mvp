import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class CausalEnv(gym.Env):
    def __init__(self, data, alpha, beta, gamma):
        super(CausalEnv, self).__init__()
        self.data = data.copy()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_step = 0
        self.action_space = spaces.Discrete(2)  # 0 = untreated, 1 = treated
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # [unit_idx, confounder, treatment]
        self.max_steps = len(data)
        
        # Print original confounder stats
        print("\nOriginal confounder statistics:")
        print(self.data['confounder'].describe())
        
        # Normalize confounders with robust scaling
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Avoid boundary values
        self.data['confounder_normalized'] = scaler.fit_transform(self.data[['confounder']]).ravel()
        
        # Print normalized confounder stats
        print("\nNormalized confounder statistics:")
        print(self.data['confounder_normalized'].describe())

    def reset(self, seed=None):
        self.current_step = 0
        unit = self.data.iloc[0]
        self.state = np.array([0, unit['confounder_normalized'], unit['treatment']], dtype=np.float32)
        return self.state, {}

    def compute_counterfactual(self, unit_idx, treatment):
        unit = self.data.iloc[unit_idx]
        confounder = unit['confounder_normalized']
        
        # Add small stochastic noise to make learning more robust
        noise = np.random.normal(0, 0.1)  # Small Gaussian noise
        
        # Compute base counterfactual
        residual = unit['outcome'] - (self.alpha + self.beta * unit['treatment'] + self.gamma * confounder)
        counterfactual_outcome = self.alpha + self.beta * treatment + self.gamma * confounder + residual
        
        # Add noise and ensure outcome remains reasonable
        noisy_outcome = counterfactual_outcome + noise
        return np.clip(noisy_outcome, -10, 10)  # Clip to reasonable range

    def step(self, action):
        unit = self.data.iloc[self.current_step]
        confounder = unit['confounder_normalized']
        actual_treatment = unit['treatment']
        actual_outcome = unit['outcome']

        # Compute counterfactual outcome based on action
        counterfactual_outcome = self.compute_counterfactual(self.current_step, action)
        reward = 10 * (counterfactual_outcome - actual_outcome)  # Scaled reward to emphasize differences

        # Update state
        self.current_step = min(self.current_step + 1, self.max_steps - 1)
        next_state = np.array([self.current_step, confounder, action], dtype=np.float32)
        done = self.current_step >= self.max_steps - 1
        return next_state, reward, done, False, {}

# Load and verify data
data = pd.read_csv('synthetic_causal_data.csv')

# Debug: Check data loading
print("\nInitial Data Check:")
print("Data shape:", data.shape)
print("\nConfounder Statistics:")
print(data['confounder'].describe())
print("\nSample of raw data:")
print(data[['confounder', 'treatment', 'outcome']].head())

# Ensure confounders are numeric and handle any missing values
data['confounder'] = pd.to_numeric(data['confounder'], errors='coerce')
if data['confounder'].isna().any():
    print("\nWarning: Found NaN values in confounder!")
    data['confounder'] = data['confounder'].fillna(data['confounder'].mean())

# Verify confounder range
print("\nConfounder range check:")
print("Min:", data['confounder'].min())
print("Max:", data['confounder'].max())

# Fit linear model
X = data[['treatment', 'confounder']]
y = data['outcome']
model_lr = LinearRegression()
model_lr.fit(X, y)
alpha = model_lr.intercept_
beta = model_lr.coef_[0]
gamma = model_lr.coef_[1]
print(f"\nLinear Model Parameters - alpha: {alpha:.4f}, beta: {beta:.4f}, gamma: {gamma:.4f}")

# Create environment with verified data
env = CausalEnv(data, alpha, beta, gamma)

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.3):
        self.env = env
        self.n_units = env.max_steps
        self.n_confounder_bins = 20  # Increased number of bins for finer granularity
        self.n_actions = env.action_space.n
        
        # Initialize Q-table with small random values for better exploration
        self.q_table = np.random.uniform(low=-0.1, high=0.1, 
                                       size=(self.n_units, self.n_confounder_bins, self.n_actions))
        
        # Initialize visit counts for exploration bonus
        self.visit_counts = np.zeros((self.n_units, self.n_confounder_bins, self.n_actions))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Track learning statistics
        self.episode_rewards = []
        self.q_value_history = []
        
        # Initialize bins based on actual data distribution
        self.confounder_bins = np.percentile(
            env.data['confounder_normalized'],
            np.linspace(0, 100, self.n_confounder_bins + 1)
        )

    def discretize_confounder(self, confounder):
        confounder_bin = np.digitize(confounder, self.confounder_bins) - 1
        bin_value = min(max(0, confounder_bin), self.n_confounder_bins - 1)
        return bin_value

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        unit_idx, confounder, _ = state
        confounder_bin = self.discretize_confounder(confounder)
        return np.argmax(self.q_table[int(unit_idx), confounder_bin])

    def learn(self, state, action, reward, next_state):
        unit_idx, confounder, _ = state
        next_unit_idx, next_confounder, _ = next_state
        
        # Ensure indices are valid
        unit_idx = int(unit_idx)
        confounder_bin = self.discretize_confounder(confounder)
        next_confounder_bin = self.discretize_confounder(next_confounder)
        
        # Add exploration bonus for less-visited states
        visit_count = self.visit_counts[unit_idx, confounder_bin, action]
        exploration_bonus = np.sqrt(1.0 / (visit_count + 1))
        
        # Update visit counts
        self.visit_counts[unit_idx, confounder_bin, action] += 1
        
        # Q-learning update with exploration bonus
        current_q = self.q_table[unit_idx, confounder_bin, action]
        next_max_q = np.max(self.q_table[int(next_unit_idx), next_confounder_bin])
        
        # Compute new Q-value with exploration bonus
        new_q = current_q + self.learning_rate * (
            reward + exploration_bonus + 
            self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table with clipping to prevent extreme values
        self.q_table[unit_idx, confounder_bin, action] = np.clip(new_q, -100, 100)

# Initialize agent
agent = QLearningAgent(env)

# Training loop with enhanced monitoring
n_episodes = 200  # Increased episodes for better learning
rewards_history = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    episode_rewards = []
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        episode_rewards.append(reward)
        
        # Debug information for confounder discretization
        if episode % 20 == 0 and not done:
            unit_idx, confounder, _ = state
            bin_idx = agent.discretize_confounder(confounder)
            print(f"\nStep {env.current_step} Debug:")
            print(f"Confounder value: {confounder:.3f}")
            print(f"Assigned bin: {bin_idx}")
            print(f"Q-values for this state: {agent.q_table[int(unit_idx), bin_idx]}")
            print(f"Reward: {reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    rewards_history.append(avg_reward)
    
    if episode % 20 == 0:
        print(f"\nEpisode {episode} completed")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Epsilon: {agent.epsilon}")
        
        # Decay epsilon for better exploitation
        agent.epsilon = max(0.01, agent.epsilon * 0.95)

# Extract optimal policy
optimal_actions = np.argmax(agent.q_table, axis=2)
print("Optimal actions per unit and confounder bin:", optimal_actions)

