import random
import json
import gym
from gym import spaces
import numpy as np
import math
from .WarehouseGraph import WarehouseGraph
import pandas as pd
import matplotlib.pyplot as plt
gym.logger.set_level(40)

# Check env/Grid.pdf for Grid details
GRID_SIZE = 7
DEPTH = math.ceil(math.sqrt(GRID_SIZE))


class WarehouseEnv(gym.Env):
    """
    Description:
        A warehouse is built in the form of a 7 x 7 grid, where each location in the grid
        can be used to store a package. Only one package can be stored at each location. 
        It becomes more expensive to store and retreive packages as we go deeper inside
        the grid. 
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

        # ------ DOUBT ------
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)

        # Warehouse Observation Space
        self.observation_space = spaces.Box(
            low=1, high=GRID_SIZE * GRID_SIZE, shape=(GRID_SIZE * GRID_SIZE, 4), dtype=np.float32)
        # ------ DOUBT ------

    def _next_observation(self):
        obs = self.current_step
        return obs

    def _take_action(self, action):
        """
        action[0] = index of item
        action[1] = deposit / retrieve ('1' or '0') 
        """
        self.itemIndex = action[0]
        self.actionType = action[1]
        self.packageID = action[2]
        self.weight = action[3]

        self.deposit = True

        if (self.actionType == 1):
            # Deposit
            if(self.current_step[self.itemIndex-1][1] == 1):
                self.deposit = False
                print("--------------------------------")
                print(
                    f'A package is already present at the location {self.current_step[self.itemIndex-1][0]}')
                print("--------------------------------")
            else:
                self.current_step[self.itemIndex-1][1] = 1
                self.current_step[self.itemIndex-1][2] = self.packageID
                self.current_step[self.itemIndex-1][3] = self.weight
        else:
            # Withdraw
            if(self.current_step[self.itemIndex-1][1] == 0):
                self.deposit = False
                print("--------------------------------")
                print(
                    f'No package is present at the location {self.current_step[self.itemIndex-1][0]}')
                print("--------------------------------")
            else:
                self.current_step[self.itemIndex-1][1] = 0
                self.current_step[self.itemIndex-1][2] = 0
                self.current_step[self.itemIndex-1][3] = 0

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

        # Episode Finish Condition
        warehouseFull = True
        for i in range(GRID_SIZE * GRID_SIZE):
            if(self.current_step[i][3] == 0):
                warehouseFull = False

        done = warehouseFull

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.depth = DEPTH
        self.grid_size = GRID_SIZE

        self.current_step = np.zeros(shape=(GRID_SIZE * GRID_SIZE, 4))

        for i in range(GRID_SIZE * GRID_SIZE):
            # [index, empty/occupied (0/1), packageID, weight]
            self.current_step[i] = (i+1, 0, 0, 0)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if(self.deposit == True):
            if(self.actionType == 0):
                actionPerformed = "Package Widthdrawn"
            else:
                actionPerformed = "Package Inserted"

            print("--------------------------------")
            print(f'Position: {self.itemIndex}')
            print(f'Package ID: {self.packageID}')
            print(f'Package Weight: {self.weight}')
            print(f'Action: {actionPerformed}')

            self.visualization = WarehouseGraph(self.current_step, "ok")

            # To view the Environment State at each step, uncomment this line
            # print(f'Step: \n {self.current_step}')
            print("--------------------------------")
