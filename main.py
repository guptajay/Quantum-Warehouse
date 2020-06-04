# Setting the path to include the yet uninstalled environment package
import argparse
from policies import MaskingMaxBoltzmannQPolicy
from agents import MaskingDQNAgent
import warehouse
from models import models
from ExperimentManager import getManagerFromConfig
from rl import memory as rl_memory
from rl import policy as rl_policy
from rl.agents.dqn import DQNAgent
import gym
from keras.models import load_model as keras_load_model
from keras.callbacks import TensorBoard
from keras import optimizers
import numpy as np
import sys
import os
package_path = os.path.join(os.getcwd(), 'environment package', 'warehouse')
sys.path.append(package_path)
# print(sys.path)

# Importing packages


manager = getManagerFromConfig('config.json')


np.random.seed(0)

# Argument parser to run easily on gg colab
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--gg_colab", type=bool, default=None,
                    help="Remove render for gg colab uses")
parser.add_argument("--do_testing", type=bool, default=None)

args = parser.parse_args()

agents = {"DQNAgent": DQNAgent}


@manager.command
def basic_run(agent, load_model, architecture, training, optimization, memory, policy, testing, do_training):

    env_name = "warehouse-v0"
    env = gym.make(env_name)

    # Overwrite the config variables with parse args
    if args.gg_colab is not None:
        training['visualize'], testing["render"] = (args.gg_colab,)*2
    if args.do_testing is not None:
        testing["do"] = args.do_testing

    # make the manager available in the environment
    env.manager = manager  # used for metrics logging by the callbacks

    # The model
    nb_actions = env.action_space.n
    input_shape = env.OS.input_shape()
    if load_model is None:
        model = getattr(models, architecture)(
            input_shape, nb_actions, use_bias=False)
    else:
        model = keras_load_model(manager.get_load_path(load_model, 'model.h5'))
    model.build(input_shape)
    print(model.summary())

    # The memory
    memory = getattr(rl_memory, memory["name"])(
        *memory["args"], **memory["kwargs"])

    # The policy
    if policy['name'] == 'MaskingMaxBoltzmannQPolicy':
        policy = MaskingMaxBoltzmannQPolicy(
            *policy["args"], **policy["kwargs"])
    else:
        policy = getattr(rl_policy, policy["name"])(
            *policy["args"], **policy["kwargs"])

    # The agent
    if agent['name'] == 'MaskingDQNAgent':
        dqn = MaskingDQNAgent(model=model, environment=env, nb_actions=nb_actions, memory=memory,
                              test_policy=policy, policy=policy,
                              *agent["args"], **agent['kwargs'])
    else:
        dqn = agents[agent["name"]](model=model, nb_actions=nb_actions,
                                    memory=memory, policy=policy, *agent["args"], **agent['kwargs'])
    dqn.compile(getattr(optimizers, optimization["optimizer"])(
        lr=optimization["lr"]), metrics=['mae'])

    # # Training
    if do_training:
        history = dqn.fit(env, verbose=2, callbacks=env.callbacks, **training)
        # TODO: Debug verbose=2, error with TrainEpisodeLogger

        # Post-training
        manager.save(dqn.model, "model")
        print("model saved")

#    Finally, evaluate our algorithm for 5 episodes.
    if testing["do"]:
        print("\nTesting the agent")
        dqn.test(env, nb_episodes=testing["nb_episodes"], visualize=testing["render"], nb_max_episode_steps=testing[
            "nb_max_episode_steps"], nb_max_start_steps=testing[
            "nb_max_start_steps"], callbacks=env.callbacks)


if __name__ == "__main__":

    manager.run('basic_run')
