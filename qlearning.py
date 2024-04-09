import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import aisd_examples
import warnings

# Ignore deprecation and user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define a function to run the training loop with specified hyperparameters
def train_agent(episodes, gamma, epsilon, decay):
    env = gym.make("aisd_examples/BlockWorld-v0", render_mode="human")

    # Create Q-table with random values and changing to list with env.observation_space.n as num_states
    qtable = np.random.rand(env.observation_space.n, env.action_space.n).tolist()

    # To store episode returns and steps per episode
    episode_returns = []
    steps_per_episode = []

    # Training loop
    for i in range(episodes):
        state_box, info = env.reset()
        state = state_box
        steps = 0
        episode_return = 0
        done = False
        
        print("episode #", i+1, "/", episodes)

        while not done:
            
            env.render()
            
            # Increment steps
            steps += 1

            # Exploration-exploitation trade-off
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = qtable[state].index(max(qtable[state]))

            # Take action
            next_state_box, reward, done, _, info = env.step(action)
            next_state = next_state_box

            # Update Q-table using Bellman equation
            qtable[state][action] = reward + gamma * max(qtable[next_state])

            # Update state and episode return
            state = next_state
            episode_return += reward

        # Decay epsilon
        epsilon -= decay * epsilon

        # Append episode return and steps per episode to lists
        episode_returns.append(episode_return)
        steps_per_episode.append(steps)
        
        print("\nDone in", steps, "steps")
        print("\nEpisode return:", episode_return)

    env.close()

    return episode_returns, steps_per_episode

# Hyperparameters for four sets
hyperparameters = [
    {"episodes": 20, "gamma": 0.1, "epsilon": 0.08, "decay": 0.1},
    {"episodes": 20, "gamma": 0.5, "epsilon": 0.05, "decay": 0.05},
    {"episodes": 20, "gamma": 0.3, "epsilon": 0.1, "decay": 0.2},
    {"episodes": 20, "gamma": 0.2, "epsilon": 0.2, "decay": 0.15}
]

# Run training loop with each set of hyperparameters
plt.figure(figsize=(12, 8))

for i, params in enumerate(hyperparameters):
    print(f"\nTraining with hyperparameters set {i+1}")
    episode_returns, steps_per_episode = train_agent(**params)
    
    hyperparameters = f"episodes: {params['episodes']}, gamma: {params['gamma']}, epsilon: {params['epsilon']}, decay: {params['decay']}"
    # Plot episode returns and steps per episode
    plt.subplot(2, 1, 1)
    plt.plot(range(1, params["episodes"] + 1), episode_returns, label=f"Parameters {hyperparameters}")
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, params["episodes"] + 1), steps_per_episode, label=f"Parameters {hyperparameters}")
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

# Add legend
plt.subplot(2, 1, 1)
plt.legend()
plt.subplot(2, 1, 2)
plt.legend()

plt.tight_layout()
plt.savefig(f"plots/v0/hyperparameters_set_plot.png")
plt.show()

