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
from keras.models import load_model as keras_load_model
import gym

from rl.agents.dqn import DQNAgent
from rl import policy as rl_policy
from rl import memory as rl_memory

from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('config.json')

from models import models
import warehouse

import argparse

np.random.seed(0)

# Argument parser to run easily on gg colab
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--gg_colab", type=bool, default=None, help="Remove render for gg colab uses")
parser.add_argument("--do_testing", type=bool, default=None)

args = parser.parse_args()

agents = {"DQNAgent":DQNAgent}

@manager.command
def basic_run(agent, load_model, architecture, training, optimization, memory, policy, testing):

    env_name = "warehouse-v0"
    env = gym.make(env_name)

    # Overwrite the config variables with parse args
    if args.gg_colab is not None:
        training['visualize'], testing["render"] = (args.gg_colab,)*2
    if args.do_testing is not None:
        testing["do"] = args.do_testing

    # make the manager available in the environment
    env.manager = manager # used for metrics logging by the callbacks

    # The model
    nb_actions = env.action_space.n
    input_shape = env.OS.input_shape()
    if load_model is None:
        model = getattr(models,architecture)(input_shape, nb_actions)
    else:
        model = keras_load_model(manager.get_load_path(load_model,'model.h5'))
    print(model.summary())

    # The memory
    memory = getattr(rl_memory,memory["name"])(*memory["args"],**memory["kwargs"])

    # The policy
    policy = getattr(rl_policy,policy["name"])(*policy["args"],**policy["kwargs"])

    # The agent
    dqn = agents[agent["name"]](model=model, nb_actions=nb_actions, memory=memory, policy=policy, *agent["args"], **agent['kwargs'])
    dqn.compile(getattr(optimizers,optimization["optimizer"])(lr=optimization["lr"]), metrics=['mae'])

    print(dqn.model.layers)
    # Log weights
    for layer in dqn.model.layers:
        weights = layer.get_weights()
        if weights:
            manager.log_histogram('{}_weights'.format(layer.name), weights[0].reshape(-1))
            manager.log_histogram('{}_biais'.format(layer.name), weights[1].reshape(-1))

    # # Training
    history = dqn.fit(env, verbose=2, callbacks=env.callbacks, **training)
    # TODO: Debug verbose=2, error with TrainEpisodeLogger



    # Post-training
    manager.save(dqn.model,"model")

#    Finally, evaluate our algorithm for 5 episodes.
    if testing["do"]:
        print("\nTesting the agent")
        dqn.test(env, nb_episodes=testing["nb_episodes"], visualize=testing["render"], callbacks=env.callbacks)

if __name__ == "__main__":

    manager.run('basic_run')



