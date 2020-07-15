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

# Check env/warehouse_grid.pdf for Grid details
GRID_SIZE = 7
DEPTH = math.ceil(math.sqrt(GRID_SIZE))

# Timesteps after which a package will be withdrawn according to normal distribution defined below
WITHDRAW_TIME = 5


class WarehouseEnv(gym.Env):
    """
    Description:
        A warehouse is built in the form of a 7 x 7 grid, where each location in the grid
        can be used to store a package. Only one package can be stored at each location. 
        It becomes more expensive to store and retreive packages as we go deeper inside
        the grid. 

    Source:
        This environment corresponds to a version of Quantum Warehouse.

    Observation:
        Type: Box(3)
        Num      Observation                                Min        Max
        0        Index (Location) in Warehouse              1          GRID_SIZE * GRID_SIZE
        1        Status of Occupancy                        0 (Vacant) 1 (Occupied)
        2        Package ID                                 20         80

    Actions:
        Type: Discrete(GRID_SIZE * GRID_SIZE)
        Num                     Action
        1                       Insert package at location 1
        2                       Insert package at location 2
        ..                      ..  
        GRID_SIZE * GRID_SIZE   Insert package at location GRID_SIZE * GRID_SIZE

        Note: The agent can only insert a package. It is withdrawn automatically
        by the environment after a fixed number of timesteps according to a
        defined normal distribution.

    Reward:
        Reward is -1 per depth level in the warehouse grid.

    Starting State:
        The warehouse is empty. 

    Episode Termination:
        There are no more packages to insert in the warehouse.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, totalPackages):
        super(WarehouseEnv, self).__init__()
        self.totalPackages = totalPackages
        self.packagesProcessed = 0

        # Reward Range
        self.reward_range = (-(DEPTH+1), -1)

        """
        Actions are discrete values of every position a package can be kept.

        Assumptions
        1. Only one package can be kept at a shelf.
        2. Every shelf has only one level. 
        """
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)

        # Warehouse Observation Space
        self.observation_space = spaces.Box(
            low=1, high=GRID_SIZE * GRID_SIZE, shape=(GRID_SIZE * GRID_SIZE, 3), dtype=np.float32)

    def _next_observation(self):
        obs = self.current_step
        return obs

    def _take_action(self, action):
        self.timestep += 1

        # action = index of space in warehouse
        self.index = action[0]
        self.packageID = action[1]

        self.withdrawFlag = False
        self.withdrawPos = 0

        # Withdraw a package from the warehouse after fixed timesteps according
        # to a normal distribution defined below
        if(self.timestep >= WITHDRAW_TIME):
            self.timestep = 0

            withdrawPackageID = math.floor(np.random.normal(50, 7))

            if(withdrawPackageID > 80):
                withdrawPackageID = 80
            if(withdrawPackageID < 20):
                withdrawPackageID = 20

            for i in range(GRID_SIZE * GRID_SIZE):
                if(self.current_step[i][2] == withdrawPackageID):
                    self.withdrawPackageID = self.current_step[i][2]
                    self.current_step[i][1] = 0
                    self.current_step[i][2] = 0
                    self.withdrawPos = i + 1
                    self.withdrawFlag = True
                    break

        # Insert a package
        self.current_step[self.index-1][1] = 1
        self.current_step[self.index-1][2] = self.packageID

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # Setting reward based on our Grid Layout
        if(1 <= self.index <= 24):
            reward = -1
        elif(25 <= self.index <= 40):
            reward = -2
        elif(41 <= self.index <= 48):
            reward = -3
        else:
            reward = -4

        # Setting reward if a package is withdrawn
        if(self.withdrawFlag):
            if(1 <= self.withdrawPos <= 24):
                reward = reward - 1
            elif(25 <= self.withdrawPos <= 40):
                reward = reward - 2
            elif(41 <= self.withdrawPos <= 48):
                reward = reward - 3
            else:
                reward = reward - 4

        # For logging purposes
        self.totalReward = reward

        # Episode Finish Condition - There are no more packages to insert
        done = False
        self.packagesProcessed += 1
        if(self.packagesProcessed >= self.totalPackages):
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.depth = DEPTH
        self.grid_size = GRID_SIZE
        self.timestep = 0

        self.current_step = np.zeros(shape=(GRID_SIZE * GRID_SIZE, 3))

        for i in range(GRID_SIZE * GRID_SIZE):
            # [index, empty/occupied (0/1), packageID]
            self.current_step[i] = (i+1, 0, 0)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("--------------------------------")
        print(f'Position: {self.index}')
        print(f'Action: Package Inserted')
        print("--------------------------------")
        if(self.withdrawFlag):
            print("")
            print("--------------------------------")
            print(f'Package ID: {self.withdrawPackageID}')
            print(f'Position: {self.withdrawPos}')
            print(f'AUTO: Package Withdrawn')
            print("--------------------------------")
        print("")
        print(f'Total Reward: {self.totalReward}')
        print("################################")

        self.visualization = WarehouseGraph(self.current_step)

        # To view the Environment State at each step, uncomment this line
        # print(f'Step: \n {self.current_step}')
