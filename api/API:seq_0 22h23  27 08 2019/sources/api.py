# Setting the path to include the yet uninstalled environment package
import sys
import os
package_path = os.path.join(os.getcwd(),'environment package','warehouse')
sys.path.append(package_path)
# print(sys.path)

# Importing packages
import os
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import load_model as keras_load_model
import gym

from rl.agents.dqn import DQNAgent
from rl import policy as rl_policy
from rl import memory as rl_memory

from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('api_config.json')

from models import models
import warehouse
from agents import MaskingDQNAgent
from policies import MaskingMaxBoltzmannQPolicy

import argparse

np.random.seed(0)

agents = {"DQNAgent":DQNAgent}

@manager.command
def basic_sequence(agent, optimization, memory, policy, load_model, testing):

    # Build environmnent
    env_name = "warehouse-v0"
    env = gym.make(env_name)

    # make the manager available in the environment
    env.manager = manager  # used for metrics logging by the callbacks

    # Load model and compile
    nb_actions = env.action_space.n
    input_shape = env.OS.input_shape()
    model = keras_load_model(manager.get_load_path(load_model,'model.h5'))

    model.build(input_shape)
    print(model.summary())

    # The memory
    memory = getattr(rl_memory,memory["name"])(*memory["args"],**memory["kwargs"])

    # The policy
    if policy['name'] == 'MaskingMaxBoltzmannQPolicy':
        policy = MaskingMaxBoltzmannQPolicy(*policy["args"],**policy["kwargs"])
    else:
        policy = getattr(rl_policy,policy["name"])(*policy["args"],**policy["kwargs"])

    # The agent
    if agent['name'] == 'MaskingDQNAgent':
        dqn = MaskingDQNAgent(model=model, environment=env,nb_actions=nb_actions, memory=memory, test_policy=policy,
                              *agent["args"], **agent['kwargs'])
    else:
        dqn = agents[agent["name"]](model=model, nb_actions=nb_actions, memory=memory, policy=policy, *agent["args"], **agent['kwargs'])
    dqn.compile(getattr(optimizers,optimization["optimizer"])(lr=optimization["lr"]), metrics=['mae'])

    dqn.test(env, nb_episodes=testing["nb_episodes"], nb_max_episode_steps=testing[
        "nb_max_episode_steps"], nb_max_start_steps=testing[
        "nb_max_start_steps"], visualize=testing["render"],
             callbacks=env.callbacks)


if __name__ == "__main__":
    manager.run('basic_sequence')