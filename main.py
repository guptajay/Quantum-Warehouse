import gym
import json
import datetime as dt

from stable_baselines.common.vec_env import DummyVecEnv
from env.WarehouseEnv import WarehouseEnv

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: WarehouseEnv()])

obs = env.reset()

print("")

# Inserting package at Index 1
action = [[1, 1]]
obs, rewards, done, info = env.step(action)
env.render()

# Inserting package at Index 20
action = [[20, 1]]
obs, rewards, done, info = env.step(action)
env.render()

# Withdrawing package at Index 1
action = [[1, 0]]
obs, rewards, done, info = env.step(action)
env.render()

print("")
