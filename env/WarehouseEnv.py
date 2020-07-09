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
WITHDRAW_TIME = 6


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
        Type: Box([Index, Status])
        Num (Index)                     Observation (Status)              
        0 - GRID_SIZE * GRID_SIZE       Empty/Occupied (0/1)  

    Actions:
        Type: Discrete(GRID_SIZE * GRID_SIZE)
        Num         Action
        n           Insert package at position n

        Note: The agent can only insert a package. It is withdrawn automatically
        by the environment for the purposes of training.

    Reward:
        Reward is -1 per depth level in the warehouse grid.

    Starting State:
        The warehouse is empty. 

    Episode Termination:
        The warehouse is full. 
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
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)

        # Warehouse Observation Space
        self.observation_space = spaces.Box(
            low=1, high=GRID_SIZE * GRID_SIZE, shape=(GRID_SIZE * GRID_SIZE, 3), dtype=np.float32)

    def _next_observation(self):
        obs = self.current_step
        return obs

    def _take_action(self, action):
        # action = index of space in warehouse
        self.index = action

        self.withdrawFlag = False
        self.withdrawPos = 0

        # Increment timestep if shelf is occupied
        for i in range(GRID_SIZE * GRID_SIZE):
            if(self.current_step[i][1] == 1):
                self.current_step[i][2] = self.current_step[i][2] + 1

             # Withdraw is package is been there for too long (for training only)
                if(self.current_step[i][2] >= WITHDRAW_TIME):
                    self.current_step[i][1] = 0
                    self.current_step[i][2] = 0
                    self.withdrawPos = i+1
                    self.withdrawFlag = True

        self.current_step[self.index-1][1] = 1

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

        if(self.withdrawFlag == True):
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

        # Episode Finish Condition - The warehouse is full
        warehouseFull = True
        for i in range(GRID_SIZE * GRID_SIZE):
            if(self.current_step[i][1] == 0):
                warehouseFull = False

        done = warehouseFull
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.depth = DEPTH
        self.grid_size = GRID_SIZE

        self.current_step = np.zeros(shape=(GRID_SIZE * GRID_SIZE, 3))

        for i in range(GRID_SIZE * GRID_SIZE):
            # [index, empty/occupied (0/1), timestep]
            self.current_step[i] = (i+1, 0, 0)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if(self.withdrawFlag == True):
            print("")
            print("--------------------------------")
            print(f'Position: {self.withdrawPos}')
            print(f'AUTO: Package Withdrawn')
            print("--------------------------------")

        print("--------------------------------")
        print(f'Position: {self.index}')
        print(f'Action: Package Inserted')
        print("--------------------------------")
        print("")
        print(f'Total Reward: {self.totalReward}')
        print("################################")

        self.visualization = WarehouseGraph(self.current_step)

        # To view the Environment State at each step, uncomment this line
        # print(f'Step: \n {self.current_step}')
