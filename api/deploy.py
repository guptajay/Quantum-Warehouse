"""
Implementation of CRUD operations for the RESTful API
"""
import gym
from WarehouseEnv import WarehouseEnv
from random import randint
import random

env = WarehouseEnv(1)
obs = env.reset()


class DeployWarehouse():
    def insertPackage(self, packageID):
        action = randint(1, 49)
        obs, rewards, done, info = env.step([action, packageID])
        return action

    def reset(self):
        obs = env.reset()
        return 'Warehouse has been successfully reset'
