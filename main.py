import gym
from env.WarehouseEnv import WarehouseEnv
import pymongo
from pymongo import MongoClient
from data.config import USERNAME, PASSWORD, DATABASE_NAME
from random import randint
import random
from pprint import pprint
import numpy as np

# Code Sample with Database Configuration

"""
## Database Configuration ##
connectionURL = "mongodb+srv://" + USERNAME + ":" + PASSWORD + \
    "@m0-cluster.vxrgp.mongodb.net/" + DATABASE_NAME + "?retryWrites=true&w=majority"

cluster = MongoClient(connectionURL)
db = cluster["warehouse"]
collection = db["schedule"]
## Database Configuration ##

totalPackages = collection.count_documents({})
packages = []

for package in collection.find():
    packages.append((package['packageID'], package['weight'], package['type']))

env = WarehouseEnv(totalPackages)
obs = env.reset()

## RL ##
done = False
epochs = 0
totalReward = 0

while not done:
    action = randint(1, 49)
    obs, rewards, done, info = env.step(
        [action, packages[epochs][0], packages[epochs][1], packages[epochs][2]])
    env.render(log=True, render=True)
    # prints the observation nicely in a dictionary
    # pprint(obs)
    totalReward += rewards
    epochs += 1

env.close()

print("")
print("Timesteps taken: {}".format(epochs))
print("Reward incurred: {}".format(totalReward))
"""

# Random Policy


def getPackageType():
    """
    Package Type    Deposit Probability
        5                   40%
        6                   30%
        7                   20%
        8                   10%
    """
    return np.random.choice(np.arange(5, 9), p=[0.4, 0.3, 0.2, 0.1])


NUM_STEPS = 1000
env = WarehouseEnv(NUM_STEPS)
obs = env.reset()

done = False
epochs = 0
totalReward = 0

while not done:
    # take a random action
    action = randint(1, 49)

    # Check if the action will overwrite a package in existing location
    validAction = False
    while not validAction:
        # take a random action again if a package is already there
        if(obs[action]['status']):
            action = randint(1, 49)
        else:
            validAction = True

    # action = {insertPos, PackageID, PackageWeight, PackageType}
    obs, rewards, done, info = env.step(
        [action, randint(1, 100), randint(1, 100), getPackageType()])
    env.render(log=True, render=True)

    totalReward += rewards
    epochs += 1

env.close()

print("")
print("Timesteps taken: {}".format(epochs))
print("Reward incurred: {}".format(totalReward))
