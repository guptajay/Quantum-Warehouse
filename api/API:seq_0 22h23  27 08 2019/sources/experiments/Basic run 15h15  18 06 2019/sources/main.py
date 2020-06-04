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
import gym

from rl.agents.dqn import DQNAgent
from rl import policy as rl_policy
from rl import memory as rl_memory

from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('config.json')

from models import models
import warehouse

np.random.seed(0)

agents = {"DQNAgent":DQNAgent}

@manager.command
def basic_run(agent, architecture, training, optimization, memory, policy, testing):

    env_name = "warehouse-v0"
    env = gym.make(env_name)

    # make the manager available in the environment
    env.manager = manager # used for metrics logging by the callbacks

    # The model
    nb_actions = env.action_space.n
    input_shape = env.OS.input_shape()
    model = getattr(models,architecture)(input_shape, nb_actions)
    print(model.summary())

    # The memory
    memory = getattr(rl_memory,memory["type"])(limit=memory["limit"], window_length=1)
    
    # The policy
    policy = getattr(rl_policy,policy)()
    
    # The agent
    dqn = agents[agent["type"]](model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=agent["nb_steps_warmup"], target_model_update=agent["target_model_update"], policy=policy)
    dqn.compile(getattr(optimizers,optimization["optimizer"])(lr=optimization["lr"]), metrics=['mae'])

    # Training
    history = dqn.fit(env, nb_steps=training["nb_steps"], nb_max_episode_steps=training["nb_max_episode_steps"], action_repetition=training["action_repetition"], verbose=0, visualize=training["render"], callbacks=env.callbacks)
    # TODO: Debug verbose=2, error with TrainEpisodeLogger

    # Post-training
    manager.save(dqn.model,"model")

    # Finally, evaluate our algorithm for 5 episodes.
    if testing["do"]:
        print("\nTesting the agent")
        dqn.test(env, nb_episodes=testing["nb_episodes"], visualize=testing["render"], callbacks = env.callbacks)
    
if __name__ == "__main__":

    manager.run('basic_run')



