import random
import json
import gym
from gym import spaces
import numpy as np
import math
from .WarehouseGraph import WarehouseGraph
gym.logger.set_level(40)

# Check env/warehouse_grid.pdf for Grid details
GRID_SIZE = 7
DEPTH = math.ceil(math.sqrt(GRID_SIZE))


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
        Type: Box(5)
        Num      Observation                                Min        Max
        0        Index (Location) in Warehouse              1          GRID_SIZE * GRID_SIZE
        1        Status of Occupancy                        0 (Vacant) 1 (Occupied)
        2        Package ID                                 20         80
        3        Package Weight                             -Inf       Inf
        4        Package Type                               1          26

    Actions:
        Type: Discrete(GRID_SIZE * GRID_SIZE)
        Num                     Action
        1                       Insert package at location 1
        2                       Insert package at location 2
        ..                      ..  
        GRID_SIZE * GRID_SIZE   Insert package at location GRID_SIZE * GRID_SIZE

        Note: The agent can only insert a package. It is withdrawn automatically
        by the environment after a normally distributed number of timesteps.

    Reward:
        Reward is -1 per depth level in the warehouse grid.

    Starting State:
        The warehouse is empty. 

    Episode Termination:
        1) There are no more packages to insert in the warehouse.
        2) Warehouse is full
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
            low=1, high=GRID_SIZE * GRID_SIZE, shape=(GRID_SIZE * GRID_SIZE, 6), dtype=np.float32)

    def _next_observation(self):
        """
        Return the warehouse observation in the form of dictionary
        to the agent
        """
        obs = {}
        for i in range(self.grid_size * self.grid_size):
            obs[i+1] = {'status': int(self.current_step[i][1]), 'packageID': int(self.current_step[i][2]),
                        'packageWeight': int(self.current_step[i][3]), 'packageType': int(self.current_step[i][4])}
        return obs

    def _take_action(self, action):
        """
        Insert and Withdraws packages

        Args:
            action (list): Contains package insert location, ID, weight & type.
        """
        # action = index of space in warehouse
        self.index = action[0]
        # ID of package
        self.packageID = action[1]
        # Weight of package
        self.packageWeight = action[2]
        # Type of package
        self.packageType = action[3]
        # Withdraw time of package drawn from it's normal distribution
        self.withdrawTime = math.floor(np.random.normal(
            self.products[self.packageType][0], self.products[self.packageType][1]))

        # If normal distributed withdrawal time is less than 1, set it to 1
        if(self.withdrawTime < 1):
            self.withdrawTime = 1

        # list of all packages withdrawn at a timestep
        self.withdrawList = []

        # Withdraw a package
        for i in range(self.grid_size * self.grid_size):
            # if withdraw time is 1, then withdraw the packgage, otherwise, reduce timestep
            if(self.current_step[i][5] == 1):
                # add withdrawal index, package ID, package weight, package type to list
                self.withdrawList.append(
                    (i+1, self.current_step[i][2], self.current_step[i][3], self.current_step[i][4]))

                # reset shelve
                self.current_step[i][1] = 0
                self.current_step[i][2] = 0
                self.current_step[i][3] = 0
                self.current_step[i][4] = 0
                self.current_step[i][5] = 0

            elif(self.current_step[i][1] == 1):
                self.current_step[i][5] -= 1

        # Insert a package and its attributes
        self.current_step[self.index-1][1] = 1
        self.current_step[self.index-1][2] = self.packageID
        self.current_step[self.index-1][3] = self.packageWeight
        self.current_step[self.index-1][4] = self.packageType
        self.current_step[self.index-1][5] = self.withdrawTime

    def step(self, action):
        """
        Execute one time step within the environment

        Args:
            action (list): Contains package insert location, ID, weight & type.
        """
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
        if(self.withdrawList):
            for package in self.withdrawList:
                if(1 <= package[0] <= 24):
                    reward = reward - 1
                elif(25 <= package[0] <= 40):
                    reward = reward - 2
                elif(41 <= package[0] <= 48):
                    reward = reward - 3
                else:
                    reward = reward - 4

        # For logging purposes
        self.totalReward = reward

        # Episode Finish Condition - There are no more packages to insert
        done = False
        self.packagesProcessed += 1
        if(self.packagesProcessed >= self.totalPackages):
            print("EPISODE TERMINATE: All packages inserted")
            done = True

        # Episode Finish Condition - Warehouse is full
        warehouseFull = True
        for i in range(self.grid_size * self.grid_size):
            if(not self.current_step[i][1]):
                warehouseFull = False
                break
        if(warehouseFull):
            print("EPISODE TERMINATE: Warehouse is full")
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.depth = DEPTH
        self.grid_size = GRID_SIZE

        # [index, empty/occupied (0/1), packageID, packageWeight, packageType, withdrawalTime]
        self.current_step = np.zeros(
            shape=(self.grid_size * self.grid_size, 6))

        # Generate Normally Distributed Product Trend
        self.products = {}
        mu = 5
        sigma = 1

        for c in range(1, 27):
            self.products[c] = (mu, sigma)
            mu += 5

        for i in range(self.grid_size * self.grid_size):
            # [index, empty/occupied (0/1), packageID, packageWeight, packageType, withdrawalTime]
            self.current_step[i] = (i+1, 0, 0, 0, 0, 0)

        return self._next_observation()

    def render(self, mode='human', log=True, render=True):
        """
        Render the Environment to the Screen

        Args:
            log (bool): Logs the warehouse status in the console.
            render (bool): Renders the GUI of the warehouse.
        """

        # Render Console Logs
        if(log):
            print("")
            print("--------------------------------")
            print(f'Package ID: {self.packageID}')
            print(f'Position: {self.index}')
            print(f'Weight: {self.packageWeight}')
            print(f'Type: {self.packageType}')
            print(f'Action: Package Inserted')
            print("--------------------------------")
            if(self.withdrawList):
                for package in self.withdrawList:
                    print("")
                    print("--------------------------------")
                    print(f'Package ID: {int(package[1])}')
                    print(f'Position: {int(package[0])}')
                    print(f'Weight: {int(package[2])}')
                    print(f'Type: {int(package[3])}')
                    print(f'AUTO: Package Withdrawn')
                    print("--------------------------------")
            print("")
            print(f'Total Reward: {self.totalReward}')
            print("################################")

        # Render the GUI of the warehouse
        if(render):
            self.visualization = WarehouseGraph(self.current_step)
