import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import aisd_examples
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

# Create Q-table with random values and changing to list with env.observation_space.n as num_states
qtable = np.random.rand(env.observation_space.n, env.action_space.n).tolist()

# Hyperparameters
episodes = 20
gamma = 0.1
epsilon = 0.08
decay = 0.1

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
        time.sleep(0.05)

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
        print("rewards",reward)
        print("total_rewards",episode_return)

    # Decay epsilon
    epsilon -= decay * epsilon

    # Append episode return and steps per episode to lists
    episode_returns.append(episode_return)
    steps_per_episode.append(steps)

    print("\nDone in", steps, "steps")
    print("\nEpisode return:", episode_return)


# Plot episode returns and steps per episode
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), episode_returns)
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.tight_layout()
plt.savefig(f"ql.png")
plt.show()

env.close()
