# Import gymnasium and import the DQN model from Stable Baselines
import gymnasium as gym
import aisd_examples
from stable_baselines3 import DQN
# Create the environment with the BlocksWorld-v0 environment from aisd_examples
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
# Initialize the DQN model with MultiInputPolicy for the environment
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_blocks")
del model # remove to demonstrate saving and loading
model = DQN.load("dqn_blocks")
# Reset the environment and get initial observation and information
obs, info = env.reset()
# Enter an infinite loop for interaction with the environment
while True:
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
    obs, info = env.reset
