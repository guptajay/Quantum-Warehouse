import gym
from env.WarehouseEnv import WarehouseEnv
import pymongo
from pymongo import MongoClient
from data.config import USERNAME, PASSWORD, DATABASE_NAME
from random import randint
import random
from pprint import pprint

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
