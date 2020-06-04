from rl.callbacks import Callback
from ExperimentManager import getManager

manager = getManager()

from warehouse.envs.constants import plt

class StatisticsCallback(Callback):
    
    def __init__(self, env):
        super(StatisticsCallback).__init__()
        self.env = env
        self.describe_action = (lambda action : action) if not hasattr(env,'AS') else (lambda action : env.AS.describe(action,verbose=1))
    
    def on_step_end(self, step, logs={}):
        self.env.current_stats.update({'current_step':step,'current_episode':logs['episode'],'current_action':self.describe_action(logs['action']),'current_reward':logs['reward']})

class DescriptionCallback(Callback):

    def __init__(self, env):
        super(DescriptionCallback, self).__init__()
        self.env = env

        #Initial metrics
        # self.init_access_time = []
        # self.init_energy_cost = []

        # Final metrics
        # self.final_access_time = []
        # self.final_energy_cost = []

        # Reward
        self.average_episode_reward_list = []
        self.current_episode_reward = 0
        
        # Running metrics
        self.steps_per_epoch = []
        self.episode = 0
        self.actions = []
        self.filling_ratio_list = []
        self.do_nothing_count = 0

    def on_step_end(self, step, logs={}):
        self.current_episode_reward += self.env.current_stats["current_reward"]
        self.steps_per_epoch[-1] += 1
        self.actions += [logs['action']]
        self.filling_ratio_list += [self.env.filling_ratio()]

        # print('action_descr :', self.env.AS.describe(logs['action']), 'filling ratio :', self.env.filling_ratio(), 'action_reward :', self.env.current_stats["current_reward"])

        # Count #do-nothing
        if logs['action']==50:
            self.do_nothing_count += 1

    def on_episode_begin(self, episode, logs={}):
        # cost = self.env.compute_storage_efficiency()
        # self.init_access_time += [cost[0]]
        # self.init_energy_cost += [cost[1]]
        self.steps_per_epoch.append(0)
        self.filling_ratio_list = []
        self.do_nothing_count = 0

    def on_episode_end(self, episode, logs={}):
        # cost = self.env.compute_storage_efficiency()
        # self.final_access_time += [cost[0]]
        # self.final_energy_cost += [cost[1]]
        self.average_episode_reward_list += [self.current_episode_reward/self.steps_per_epoch[-1]]
        self.current_episode_reward = 0
        self.number_unique_episode_action = len(set(self.actions))

        print("End of episode {} with {} steps, and average episode reward {}, and unique number of actions during episode: {} and final filling ratio : {}".format(self.episode,self.steps_per_epoch[-1], self.average_episode_reward_list[-1], self.number_unique_episode_action, self.filling_ratio_list[-1]))

        if hasattr(self.env,'manager'):
            self.env.manager.log_scalars('training',[self.average_episode_reward_list[-1],self.steps_per_epoch[-1]] ,['Average reward','Number of episodes'])
            self.env.manager.log_scalars('max_filling_ratio', [max(self.filling_ratio_list)])
            self.env.manager.log_scalars('do_nothing_count', [self.do_nothing_count])
        self.episode += 1


class DebugCallback(Callback):

    def __init__(self, env):
        super(DebugCallback, self).__init__()
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        print('\nBeginning episode {}'.format(episode))

    def on_episode_end(self, episode, logs={}):
        #print("End of episode {}, with average access time : {}, and average energy cost : {}".format(episode, self.env.compute_storage_efficiency()[0], self.env.compute_storage_efficiency()[1]))
        print("End of episode {}, with average access time : {}, and average energy cost : {}".format(episode, *self.env.compute_storage_efficiency()))

    def on_action_end(self, action, logs={}):
        #print("End of action (target:{}), with average access time : {}, and average energy cost : {}".format(action, self.env.compute_storage_efficiency()[0], self.env.compute_storage_efficiency()[1]))
        print("End of action (target:{}), with average access time : {}, and average energy cost : {}".format(action, *self.env.compute_storage_efficiency()))

class HistogramCallback(Callback):

    def __init__(self):
        super(HistogramCallback, self).__init__()


    def on_step_end(self, step, logs={}):

        if step%20 == 0:
            for layer in self.model.layers:
                weights = layer.get_weights()
                if weights:
                    manager.log_histogram('{}_weights'.format(layer.name), weights[0].reshape(-1))
                    # manager.log_histogram('{}_biais'.format(layer.name), weights[1].reshape(-1))