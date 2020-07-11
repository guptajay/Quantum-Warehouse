import gym
import json
import datetime as dt
from env.WarehouseEnv import WarehouseEnv
import pymongo
from pymongo import MongoClient
from data.config import USERNAME, PASSWORD, DATABASE_NAME
from random import randint
import numpy as np
import random

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
    packages.append((package['packageID'], package['weight']))

env = WarehouseEnv(totalPackages)
obs = env.reset()

## RL ##
done = False
epochs = 0
totalReward = 0

while not done:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    print("Package ID:", packages[epochs][0])
    print("Package Weight:", packages[epochs][1])
    env.render()
    print("")
    totalReward += rewards
    epochs += 1


print("Timesteps taken: {}".format(epochs))
print("Reward incurred: {}".format(totalReward))
