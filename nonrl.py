import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import aisd_examples

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

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
    # Get the initial observation from the environment
    observation, info = env.reset()
    state = observation
    steps = 0
    episode_return = 0
    done = False
    
    print("Episode #", i+1, "/", episodes)

    while not done:
        # Render the environment (optional)
        env.render()
        time.sleep(0.05)

        # Increment steps
        steps += 1

        # Calculate the distance from the red ball position to the center of the image
        distance_to_center = abs(observation - 320)

        # Choose action based on the distance to the ball position
        if distance_to_center < 100:  # Move towards the ball if it's within a certain range
            action = observation + 50
        elif distance_to_center < 200:  # Move towards the ball if it's within a certain range
            action = observation + 150
        else:  # Move towards the center of the image if the ball is far away
            action = 350

        # Take the action in the environment
        next_state_box, reward, done, _, info = env.step(action)
        next_state = next_state_box

        # Update state and episode return
        state = next_state
        episode_return += reward

    # Append episode return and steps per episode to lists
    episode_returns.append(episode_return)
    steps_per_episode.append(steps)

    print("\nDone in", steps, "steps")
    print("Episode return:", episode_return)

# Plot episode returns and steps per episode
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), episode_returns)
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.tight_layout()
plt.savefig(f"nonrl.png")
plt.show()

env.close()

