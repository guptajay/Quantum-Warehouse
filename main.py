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

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: WarehouseEnv()])

obs = env.reset()

print("")

for package in collection.find():
    action = [randint(1, 49)]
    obs, rewards, done, info = env.step(action)

    print("Package ID:", package['packageID'])
    print("Package Weight:", package['weight'])
    env.render()
    print("")
