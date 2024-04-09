# Import gymnasium and import the DQN model from Stable Baselines
import gymnasium as gym
import aisd_examples
from stable_baselines3 import PPO
# Create the environment with the BlockWorld-v0 environment from aisd_examples
env = gym.make("aisd_examples/BlockWorldSB", render_mode="human")
# Initialize the PPO model with MultiInputPolicy for the environment
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo_blocks")
del model # remove to demonstrate saving and loading
model = PPO.load("ppo_blocks")
# Reset the environment and get initial observation and information
obs = env.reset()
# Enter an infinite loop for interaction with the environment
while True:
  action, _states = model.predict(obs)
  obs, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
     obs, _ = env.reset()
