import gymnasium as gym
import numpy as np
import aisd_examples
import matplotlib.pyplot as plt

env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
observation, info = env.reset()

episode_returns = []
episode_return = 0

# Define the maximum reward and the maximum allowed movement distance
max_reward = 100.0
max_movement = 50 


for _ in range(1000):
    terminated = False
    # Null agent's action choice based on a heuristic
    while not terminated:
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

    observation, reward, terminated, truncated, info = env.step(action)
    episode_return += reward

    # If episode terminates or truncates, reset and log episode return
    if terminated or truncated:
        observation, info = env.reset()
        episode_returns.append(episode_return)
        episode_return = 0

env.close()

# Plot episode returns
plt.plot(episode_returns)
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()

