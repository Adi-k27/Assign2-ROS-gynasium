import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import aisd_examples

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

# Hyperparameters
episodes = 10
gamma = 0.1
epsilon = 0.08
decay = 0.1

# To store episode returns and steps per episode
episode_returns = []
steps_per_episode = []

# Define the maximum reward and the maximum allowed movement distance
max_reward = 100.0
max_movement = 640

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

        # Limit the movement to only some amount
        if distance_to_center > max_movement:
            # Determine the direction to move (left or right)
            direction = -100 if observation < 320 else 100
            # Move only by the maximum allowed movement amount
            action = observation + direction * max_movement
        else:
            # Move to the center of the image
            action = 320

        # Take the action in the environment
        next_state_box, reward, done, _, info = env.step(action)
        next_state = next_state_box

        # Update Q-table (not used in this version)
        # qtable[state][action] = reward + gamma * max(qtable[next_state])

        # Update state and episode return
        state = next_state
        episode_return += reward
        print(reward)

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
plt.savefig(f"nonql.png")
plt.show()

env.close()

