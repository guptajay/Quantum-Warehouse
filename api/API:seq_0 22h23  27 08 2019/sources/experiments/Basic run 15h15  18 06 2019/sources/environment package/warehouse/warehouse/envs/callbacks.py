from rl.callbacks import Callback

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
        self.init_access_time = []
        self.init_energy_cost = []

        # Final metrics
        self.final_access_time = []
        self.final_energy_cost = []

        # Reward
        self.average_episode_reward_list = []
        self.current_episode_reward = 0
        
        # Running metrics
        self.steps_per_epoch = []
        self.episode = 0

    def on_step_end(self, step, logs={}):
        self.current_episode_reward += self.env.current_stats["current_reward"]
        self.steps_per_epoch[-1] += 1

    def on_episode_begin(self, episode, logs={}):
        cost = self.env.compute_storage_efficiency()
        self.init_access_time += [cost[0]]
        self.init_energy_cost += [cost[1]]
        self.steps_per_epoch.append(0)

    def on_episode_end(self, episode, logs={}):
        cost = self.env.compute_storage_efficiency()
        self.final_access_time += [cost[0]]
        self.final_energy_cost += [cost[1]]
        self.average_episode_reward_list += [self.current_episode_reward/self.steps_per_epoch[-1]]
        self.current_episode_reward = 0

        print("End of episode {} with {} steps, average access time : {}, average energy cost : {}, and average episode reward {}".format(self.episode,self.steps_per_epoch[-1],cost[0],cost[1], self.average_episode_reward_list[-1]))

        if hasattr(self.env,'manager'):
            self.env.manager.log_scalars('training',[self.init_access_time[-1],self.final_access_time[-1],self.init_energy_cost[-1],self.final_energy_cost[-1],self.average_episode_reward_list[-1],self.steps_per_epoch[-1]],['Initial access time','Final access time','Initial energy cost','Final energy cost','Average reward','Number of episodes'])
        
        self.episode += 1

    def plot_metrics(self):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize =(20,20))
        ax1.plot(list(range(self.episode)), self.init_access_time[:self.episode], color="g", label="Init access time")
        ax1.plot(list(range(self.episode)), self.final_access_time, color="r", label="Final access time")
        ax1.set_title("Init - Final access_time over episode")
        ax1.legend()

        ax2.plot(list(range(self.episode)), self.init_energy_cost[:self.episode], color="g", label="Init energy cost")
        ax2.plot(list(range(self.episode)), self.final_energy_cost, color="r", label="Final energy cost")
        ax2.set_title("Init - Final energy cost over episode")
        ax2.legend()

        ax3.plot(list(range(self.episode)), self.average_episode_reward_list, color="b")
        ax3.set_title("Average reward over episode")

        ax4.plot(list(range(self.episode)), self.steps_per_epoch[:self.episode], color="b")
        ax4.set_title("Steps per episode")

        # plt.show()
        return f
        

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
