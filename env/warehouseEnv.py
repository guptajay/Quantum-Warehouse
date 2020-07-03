import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

GRID_SIZE = 7
DEPTH = math.ceil(math.sqrt(GRID_SIZE))


class Warehouse(gym.Env):
    """A warehouse environment for OpenAI Gym"""

    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(Warehouse, self).__init__()

        self.df = df
        self.reward_range = (-(DEPTH+1), -1)

        obs = np.zeros(shape=(GRID_SIZE * GRID_SIZE, 2))
        for i in range(GRID_SIZE * GRID_SIZE):
            obs[i] = (i, 0)

        # Actions are discrete values of every position a package can be kept.
        # Assumed only one shelf per position.
        self.action_space = spaces.Discrete(7 * 7)

        # Warehouse current state
        self.observation_space = spaces.Box(
            low=1, high=49, shape=(7, 7), dtype=np.float16)

    def _next_observation(self):
        obs = self.current_step
        return obs

    def _take_action(self, action):
        # action[0] = index of item
        # action[1] = deposit / retrieve ('1' or '0')
        self.itemIndex = action[0]
        self.actionType = action[1]

        # Ignoring all error handling for now
        # Deposit Item
        if (actionType == 1):
            self.current_step[self.itemIndex-1][1] = 1
        else:
            self.current_step[self.itemIndex-1][1] = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if(1 <= self.itemIndex <= 24):
            reward = -1
        elif(25 <= self.itemIndex <= 40):
            reward = -2
        elif(41 <= self.itemIndex <= 48):
            reward = -3
        else:
            reward = -4

        # Random Ending (Need to figure out)
        done = self.itemIndex == 60

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.depth = DEPTH
        self.grid_size = GRID_SIZE

        self.current_step = np.zeros(shape=(GRID_SIZE * GRID_SIZE, 2))

        for i in range(GRID_SIZE * GRID_SIZE):
            self.current_step[i] = (i+1, 0)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Package ID: {self.itemIndex}')
        print(f'Action: {self.actionType}')
