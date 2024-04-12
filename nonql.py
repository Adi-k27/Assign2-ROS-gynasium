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

# Define the maximum reward and the maximum allowed movement distance
max_reward = 100.0
max_movement = 50

# Training loop
for i in range(episodes):

    # Get the current position of the red ball
    red_ball_position, info = env.reset()
    observation, info = 
    state = observation
    steps = 0
    episode_return = 0
    done = False
    
    print("episode #", i+1, "/", episodes)

    while not done:
    	# Calculate the distance from the red ball position to the center of the image
    	distance_to_center = abs(observation - 320)
    	# Calculate the reward based on the distance to the center
    	reward = max_reward - distance_to_center
    	# Limit the movement to only some amount
    	if distance_to_center > max_movement:
    	    # Determine the direction to move (left or right)
    	    direction = -1 if observation < 320 else 1
    	    # Move only by the maximum allowed movement amount
    	    action = observation + direction * max_movement
    	else: 
    	    # Move to the center of the image
    	    action = 320

    # Take the action in the environment
    observation, _, terminated, _, _ = env.step(action)
    episode_return += reward

    # Append episode return and steps per episode to lists
    episode_returns.append(episode_return)
    steps_per_episode.append(steps)

    print("\nDone in", steps, "steps")
    print("\nEpisode return:", episode_return)


# Plot episode returns and steps per episode
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(range(1, episodes + 1), episode_returns)
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(range(1, episodes + 1), steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.show()

env.close()
