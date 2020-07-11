import gym
import json
import datetime as dt
from stable_baselines.common.vec_env import DummyVecEnv
from env.WarehouseEnv import WarehouseEnv
import pymongo
from pymongo import MongoClient
from data.config import USERNAME, PASSWORD, DATABASE_NAME
from random import randint

## Database Configuration ##
connectionURL = "mongodb+srv://" + USERNAME + ":" + PASSWORD + \
    "@m0-cluster.vxrgp.mongodb.net/" + DATABASE_NAME + "?retryWrites=true&w=majority"

cluster = MongoClient(connectionURL)
db = cluster["warehouse"]
collection = db["schedule"]
## Database Configuration ##

totalPackages = collection.count_documents({})

# The algorithms require a vectorized environment to run
env = WarehouseEnv(totalPackages)

obs = env.reset()

print("")

packages = []

for package in collection.find():
    packages.append((package['packageID'], package['weight']))

done = False
epochs = 0
totalReward = 0

while not done:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)

    totalReward += rewards
    epochs += 1

    print("Package ID:", package['packageID'])
    print("Package Weight:", package['weight'])
    env.render()
    print("")


print("Timesteps taken: {}".format(epochs))
print("Reward incurred: {}".format(totalReward))
