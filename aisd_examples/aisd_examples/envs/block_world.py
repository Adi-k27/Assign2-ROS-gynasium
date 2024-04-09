# Environment file of version without target in the current state
# Importing the libraries
import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np

class BlockWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # Initialize Prolog interpreter and load the blocks world
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        
        # Verifying and loading the blocks_world prolog file that has no target within current state
        result = self.prolog_thread.query('[blocks_world]')

        # Set up dictionary to convert Prolog state names into integers
        states_result = self.prolog_thread.query("state(State)")
        # states_result will be like [{'State': 'bc1'}, {'State': 'bc2'}]
        # coverting [{'State': 'bc1'}, {'State': 'bc2'}] type of data to {'bc1': 0, 'bc2': 1} type of data
        self.states_dict = {state['State']: i for i, state in enumerate(states_result)}
        #self.states_dict will be like {'bc1': 0, 'bc2': 1} which means a is on b, b is on c and c is on 1 state

        # Set up dictionary to convert action numbers into Prolog actions
        self.actions_dict = {}
        actions_result = self.prolog_thread.query("action(A)")
        # actions_result be like [{'A': {'args': ['a', 'b', 'c'], 'functor': 'move'}}, {'A': {'args': ['a', 'b', 1], 'functor': 'move'}}]
        for i, A in enumerate(actions_result):
            action_string = A['A']['functor'] #obtains functor - move
            first = True
            for arg in A['A']['args']: #obtains args like args: ['a', 'b', 'c']
                if first:
                    first = False
                    action_string += '(' 
                else:
                    action_string += ','
                action_string += str(arg)
            action_string += ')'
            self.actions_dict[i] = action_string #appends args with functor in move(a,b,c) format and add it to action dic
        # self.actions_dict be like {0: 'move(a,b,c)', 1: 'move(a,b,1)'}

        # Define observation space
        self.observation_space = spaces.Discrete(len(self.states_dict))

        # Define action space
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # Store initial starting state and target state
        self._agent_location = 0  # Assuming the first state is the initial starting state
        self._target_location = np.random.randint(0, len(self.states_dict))  # Assuming the target state is the random here

        if self.render_mode == "human":
            self.display = Display()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
    	# Set a fixed target state so the Qlearning can be applied and table is updated and agent can learn
    	self.target = '23a'
    	# Obtaining index value of the target state
    	self._target_location = list(self.states_dict.values())[list(self.states_dict.keys()).index(self.target)]
    	
    	if self.display is not None:
            # Obtaining value of the target state to be displayed
            self.display.target = list(self.states_dict.keys())[self._target_location]

   	# Issue Prolog query to reset
    	self.prolog_thread.query("reset")

    	# Issue Prolog query to retrieve current state
    	state = self.prolog_thread.query("current_state(State)")
    	
    	# Obtaining index value of the agent state
    	self._agent_location = list(self.states_dict.values())[list(self.states_dict.keys()).index(state[0]['State'])]

    	observation = self._agent_location
    	# Info about the agent and target at every reset
    	info = { "agent_initial": state[0]['State'], "agent_target": self.target}
    	print(info)

    	return observation, info


    def step(self, action):
        
        reward = 0
        done = False
        
        # Obtaining index value of the action
        action_value = list(self.actions_dict.values())[list(self.actions_dict.keys()).index(action)]
        result = self.prolog_thread.query('step(' + action_value + ')') # Obtaining result of action

        # If action is possible then obtain the agent current state and reward -1
        if result:
            state = self.prolog_thread.query("current_state(State)") 
            self._agent_location = list(self.states_dict.values())[list(self.states_dict.keys()).index(state[0]['State'])]
            reward = -1
        else: # If action is impossible then provide reward -10
            reward = -10

        # Check if episode is done then reward is 100
        done = (self._agent_location == self._target_location)
        reward = 100 if done else reward

        observations = self._agent_location
        if self.render_mode == "human":
            self.render()

        return observations, reward, done, False,{}

    def render(self):
        if self.render_mode == "human":
            self.display.step(list(self.states_dict.keys())[self._agent_location])

    def close(self):
        if self.mqi is not None:
            self.mqi.stop()

