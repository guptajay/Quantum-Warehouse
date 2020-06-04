import numpy as np
from abc import ABC, abstractclassmethod

class RewardFunctionGenerator(ABC):
    ''' Parent class for all reward function generators.  
    '''

    def __init__(self, warehouse):
        self.spots = warehouse.spots
        self.n_spots = warehouse.n_spots
        self.queue = warehouse.queue
        self.n_queue = warehouse.n_queue
        self.action_space = warehouse.AS
        self.compute_access_time = warehouse.compute_access_time
        self.compute_access_cost = warehouse.compute_access_cost
        self.warehouse = warehouse

    @abstractclassmethod
    def compute_reward(self,action):
        ''' Compute the reward associated to an action taken by the agent.
        '''
        return 0

    @abstractclassmethod
    def compute_reward_for_retrieval(self,spot):
        ''' Compute the reward associated to retrieving a package from a spot. 
        
        This does not correspond to an action from the agent but from the environment.
        '''
        return 0

    def __call__(self,action):
        return self.compute_reward(action)

class BasicReward(RewardFunctionGenerator):
    '''
    Victor's take on basic rewards.
    '''

    # cost_factor = 0.5
    # time_factor = 2
    #
    cost_factor = 0
    time_factor = 1
    
    improvement_factor = 3
    movement_factor = 1
    
    retrieval_factor = 2

    def __init__(self, warehouse):
        super().__init__(warehouse)
        self.max_access_time = warehouse.max_access_time
        self.max_access_cost = warehouse.max_access_cost

    def compute_reward(self, action):
                
        if self.action_space.is_invalid(action):
            return -1

        if self.action_space.is_do_nothing(action):
            return -1

        if self.action_space.is_drop_package(action):
            
            target_spot_id = self.action_space.drop_to_id(action)
            target_spot = self.spots[target_spot_id]
            return self.spot_reward(target_spot)

        if self.action_space.is_move_package(action):

            target_spot_id = self.action_space.move_to_id(action)
            target_spot = self.spots[target_spot_id]

            reference_spot_id = self.action_space.move_from_id(action)
            reference_spot = self.spots[reference_spot_id]

            movement_time = self.compute_access_time(reference_spot,target_spot)
            movement_cost = self.compute_access_cost(reference_spot,target_spot)

            reference_reward = self.spot_reward(reference_spot)
            target_reward = self.spot_reward(target_spot)
            print('reference_reward : {}, target_reward : {}, movement_time : {}'.format(reference_reward, target_reward, movement_time))
            return (target_reward - reference_reward)*BasicReward.improvement_factor + (movement_time*BasicReward.time_factor + movement_cost*BasicReward.cost_factor)*BasicReward.movement_factor


    def compute_reward_for_retrieval(self,spot):
            return self.spot_reward(spot)*BasicReward.retrieval_factor


    def spot_reward(self,spot):
        ''' Support function to compute the reward of putting a package in a spot.
        '''
        access_time = spot.access_time
        access_cost = spot.access_cost()

        normalized_access_time_quality =  access_time/self.max_access_time
        normalized_access_cost_quality = access_cost/self.max_access_cost

        return normalized_access_cost_quality*BasicReward.cost_factor + normalized_access_time_quality*BasicReward.time_factor

    def episode_completion_reward(self):
        return 20

class RefinedReward(RewardFunctionGenerator):

    cost_factor = 10
    time_factor = 1

    improvement_factor = 3
    movement_factor = 1

    retrieval_factor = 2

    def __init__(self, warehouse):
        super().__init__(warehouse)
        self.max_access_time = warehouse.max_access_time
        self.max_access_cost = warehouse.max_access_cost

    def compute_reward(self, action):

        if self.action_space.is_invalid(action):
            return -1

        if self.action_space.is_do_nothing(action):
            return -1

        if self.action_space.is_drop_package(action):
            target_spot_id = self.action_space.drop_to_id(action)
            target_spot = self.spots[target_spot_id]
            return self.spot_reward(target_spot)

        if self.action_space.is_move_package(action):
            target_spot_id = self.action_space.move_to_id(action)
            target_spot = self.spots[target_spot_id]

            reference_spot_id = self.action_space.move_from_id(action)
            reference_spot = self.spots[reference_spot_id]

            movement_time = self.compute_access_time(reference_spot, target_spot)
            movement_cost = self.compute_access_cost(reference_spot, target_spot)

            reference_reward = self.spot_reward(reference_spot)
            target_reward = self.spot_reward(target_spot)
            print(
                'reference_reward : {}, target_reward : {}, movement_time : {}'.format(reference_reward, target_reward,
                                                                                       movement_time))
            return (target_reward - reference_reward)*BasicReward.improvement_factor + (
                        movement_time*BasicReward.time_factor + movement_cost*BasicReward.cost_factor)*BasicReward.movement_factor


    def compute_reward_for_retrieval(self,spot):
            return self.spot_reward(spot)*BasicReward.retrieval_factor

    def spot_reward(self,spot):
        ''' Support function to compute the reward of putting a package in a spot.
        '''
        access_time = spot.access_time
        access_cost = spot.access_cost()

        normalized_access_time_quality =  access_time/self.max_access_time -.5
        normalized_access_cost_quality = access_cost/self.max_access_cost -.5

        return normalized_access_cost_quality*BasicReward.cost_factor + normalized_access_time_quality*BasicReward.time_factor

    def episode_completion_reward(self):
        return 0

    def compute_reward_for_batch_retrieval(self, ratio=.5):
        return sum((self.compute_reward_for_retrieval(spot) for spot in np.random.choice(self.spots,
                                                                                         int(np.ceil(
                                                                                             self.n_spots*ratio)),
                                                                                         replace=True)))/int(np.ceil(
                                                                                             self.n_spots*ratio))

    # def improved_reward_for_batch_retrieval(self, ratio=.5):
    #     return sum((self.warehouse.retrieve_package(reward=True) for _ in range(self.)))

