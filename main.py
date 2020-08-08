import gym
from env.WarehouseEnv import WarehouseEnv
import pymongo
from pymongo import MongoClient
from data.config import USERNAME, PASSWORD, DATABASE_NAME
from random import randint
import random
from pprint import pprint
import numpy as np
import tensorflow as tf
import datetime
from agents.random import random_policy
from agents.baseline import baseline_policy
import sys

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

# TensorBoard Setup
current_time = datetime.datetime.now().strftime(
    sys.argv[1] + "-%Y/%m/%d-%H/%M/%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


def getPackageType():
    """
    Package Type    Deposit Probability
        8                   40%
        9                   30%
        10                  20%
        11                  10%
    """
    return np.random.choice(np.arange(8, 12), p=[0.4, 0.3, 0.2, 0.1])


for i in range(10000):
    NUM_STEPS = 100
    env = WarehouseEnv(NUM_STEPS)
    obs = env.reset()

    done = False
    epochs = 0
    totalReward = 0

    while not done:
        # take a action according to a policy passed in the program argument
        if(sys.argv[1] == 'random_policy'):
            action = random_policy(obs)
        elif(sys.argv[1] == 'baseline_policy'):
            action = baseline_policy(obs)

        # action = {insertPos, PackageID, PackageWeight, PackageType}
        obs, rewards, done, info = env.step(
            [action, randint(1, 100), randint(1, 50), getPackageType()])
        env.render(log=True, render=True)
        totalReward += rewards
        epochs += 1

    # Logging for TensorBoard
    # (1) Episode Length
    # (2) Episode Reward
    with summary_writer.as_default():
        tf.summary.scalar('Episode Length', epochs, step=i)
        tf.summary.scalar('Episode Reward', totalReward, step=i)

    env.close()

    print("")
    print("Timesteps taken: {}".format(epochs))
    print("Reward incurred: {}".format(totalReward))
    print("Episode Num: {}".format(i+1))
