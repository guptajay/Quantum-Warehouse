import random
import json
import gym
from gym import spaces
import numpy as np
import math
gym.logger.set_level(40)

# Check env/Grid.pdf for Grid details
GRID_SIZE = 7
DEPTH = math.ceil(math.sqrt(GRID_SIZE))


class WarehouseEnv(gym.Env):
    """
    A warehouse environment for OpenAI Gym.
    Reinforcement Learning architecture to automate long term planning of warehouse inventory for enterprise deployment.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(WarehouseEnv, self).__init__()

        # Reward Range
        self.reward_range = (-(DEPTH+1), -1)

        """
        Actions are discrete values of every position a package can be kept.

        Assumptions
        1. Only one package can be kept at a shelf.
        2. Every shelf has only one level. 
        """
        self.action_space = spaces.Discrete(7 * 7)

        # Warehouse Observation Space
        self.observation_space = spaces.Box(
            low=1, high=49, shape=(49, 2), dtype=np.float32)

    def _next_observation(self):
        obs = self.current_step
        return obs

    def _take_action(self, action):
        """
        action[0] = index of item
        action[1] = deposit / retrieve ('1' or '0')

        For the time being, it is assumed that the index of a package is the same as the package ID. 
        """
        self.itemIndex = action[0]
        self.actionType = action[1]

        # Deposit Item
        if (self.actionType == 1):
            self.current_step[self.itemIndex-1][1] = 1
        else:
            self.current_step[self.itemIndex-1][1] = 0

        # TODO - Error Handling

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # Setting reward based on our Grid Layout
        if(1 <= self.itemIndex <= 24):
            reward = -1
        elif(25 <= self.itemIndex <= 40):
            reward = -2
        elif(41 <= self.itemIndex <= 48):
            reward = -3
        else:
            reward = -4

        # TODO - Figure out this 'done'
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

        if(self.actionType == 0):
            actionPerformed = "Package Widthdrawn"
        else:
            actionPerformed = "Package Inserted"

        print("--------------------------------")
        print(f'Package ID: {self.itemIndex}')
        print(f'Action: {actionPerformed}')

        # To view the Environment State at each step, uncomment this line
        # print(f'Step: \n {self.current_step}')
        print("--------------------------------")
