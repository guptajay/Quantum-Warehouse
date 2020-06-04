from gym import spaces
import numpy as np
from abc import ABC, abstractclassmethod

class ActionSpace(ABC):
    ''' Parent class for all action spaces. 
    
    Since we want to integrate different action spaces in modular code, we need to define verbose functions to understand 
    any possible mapping between action_id and what the action is. This allows to define reward functions independently of action spaces.
    
    '''

    def __init__(self, warehouse):
        self.spots = warehouse.spots
        self.n_spots = warehouse.n_spots
        self.queue = warehouse.queue
        self.n_queue = warehouse.n_queue

    @abstractclassmethod
    def sample(self):
        ''' Draw a random action from the action space.
        '''
        return None
        
    def is_do_nothing(self, action):
        ''' Return True if the action id corresponds to doing nothing.
        '''
        return False
        
    def is_drop_package(self,action):
        ''' Returns True if the action corresponds to dropping a package from the dropping queue.
        '''
        return False
        
    def drop_to_id(self,action):
        ''' If action corresponds to dropping a package from the queue, this will return the spot id to which the action drops.
        '''
        if self.is_drop_package(action):
            return None
        else:
            raise Exception('Internal Error: drop_to_id was called on an action that is not a dropping one')

    def drop_from_id(self,action):
        ''' If action corresponds to dropping a package from the queue, this will return the spot id in the queue from which the action drops.
        '''
        if self.is_drop_package(action):
            return None
        else:
            raise Exception('Internal Error: drop_to_id was called on an action that is not a dropping one')
        
    def is_move_package(self,action):
        ''' Returns True if the action corresponds to moving a package from the dropping queue.
        '''
        return False
        
    def move_from_id(self,action):
        ''' If action corresponds to moving a package, this will return the spot id from which the action moves.
        '''
        if self.is_move_package(action):
            return None
        else:
            raise Exception('Internal Error: move_from_id was called on an action that is not a moving one')
        
    def move_to_id(self,action):
        ''' If action corresponds to moving a package, this will return the spot id to which the action moves.
        '''
        if self.is_move_package(action):
            return None
        else:
            raise Exception('Internal Error: move_to_id was called on an action that is not a moving one')

    
    def is_invalid(self,action):
        ''' Return True if the action is invalid. No need to overwrite.
        '''
        if self.is_do_nothing(action):
            return False

        if self.is_drop_package(action):
            return self.spots[self.drop_to_id(action)].is_full

        if self.is_move_package(action):
            return self.spots[self.move_to_id(action)].is_full or not self.spots[self.move_from_id(action)].is_full

    def describe(self,action, verbose = 0):
        ''' Verbose description of the action.
        '''
        if self.is_invalid(action):
            return 'Invalid action'

        if self.is_do_nothing(action):
            return 'Doing nothing'

        if self.is_drop_package(action):
            return 'Dropping a package' + (' from {} to {}'.format(self.drop_from_id(action),self.drop_to_id(action)) if verbose>0 else '')
                
        if self.is_move_package(action):
            return 'Moving a package' + (' from {} to {}'.format(self.move_from_id(action),self.move_to_id(action)) if verbose>0 else '')
            
        return 'Unrecognized action {}'.format(action)



class BasicAS(ActionSpace):
    '''Simplefied action space.
    
    We only allow the agent to drop the package from the queue to the one of the spots, or do nothing.
    
    '''
    
    def __init__(self, warehouse):
        super().__init__(warehouse)
        self.dot = self.n_queue*self.n_spots
        self.space = spaces.Discrete(self.n_queue*self.n_spots + 1)
        # (drop queue_0 to spot_0, drop queue_0 to spot_1, ... , drop queue_0 to spot_n , drop queue_1 to spot_0 , ... , drop queue_n to spot_n, do_nothing)

        
    def sample(self):
        return np.random.randint(0,self.dot+1)
        
    def is_do_nothing(self,action):
        return action == self.dot
        
    def is_drop_package(self,action):
        return action < self.dot
        
    def drop_to_id(self,action):
        if self.is_drop_package(action):
            return action%self.n_spots
        else:
            raise Exception('Internal Error: drop_to_id was called on an action that is not a dropping one')

    def drop_from_id(self,action):
        ''' If action corresponds to dropping a package from the queue, this will return the spot id in the queue from which the action drops.
        '''
        if self.is_drop_package(action):
            return action//self.n_spots
        else:
            raise Exception('Internal Error: drop_to_id was called on an action that is not a dropping one')
    
    