from gym import spaces
import numpy as np
from abc import ABC, abstractclassmethod

from warehouse.envs.constants import n_packages


class ObservationSpace(ABC):
    ''' Parent class for all observation spaces
    '''

    def __init__(self, warehouse):
        self.spots = warehouse.spots
        self.n_spots = warehouse.n_spots
        self.queue = warehouse.queue
        self.n_queue = warehouse.n_queue

    @abstractclassmethod
    def compute_state(self):
        ''' 
        Return the current observation based on the warehouse's spots and on the queue of objects to drop.
        ''' 
        pass
        
    @abstractclassmethod
    def input_shape(self):
        '''
        Compute the associated input shape.
        '''

    def __call__(self):
        return self.compute_state()
        
    def is_well_defined(self, observation_space, space_to_test=None):
        ''' 
        Run tests to check the validity of the observation space.
        '''
        return True

class BasicOS(ObservationSpace):
    ''' 
    The agent will only see an integer for each spot (0 if empty, 1 if filled with object of type 1 and 2 if filled with object of type 2).

    Notice that we do not give information about the frequency or weight. Since there's a bijection between the object ID and its characteristics we believe it is unnecessary at first. We also don't give information about access time, there's also a bijection with the spots.

    This basic OS should allow us to see if the agent can learn to place objects with minimal strategy.
    '''

    def __init__(self, warehouse):
        ''' By convention, the dropping queue is at the end of the observation space!
        '''
        super().__init__(warehouse)
        self.space = spaces.Tuple( (spaces.Discrete(n_packages) for _ in range(self.n_spots+self.n_queue)) )

    def compute_state(self):
        return  [  *( (spot.package.id,) if spot.is_full else (0,) for spot in self.spots ), *( (spot.package.id,) if spot.is_full else (0,) for spot in self.queue ) ]
        
    def input_shape(self):
        return (self.n_spots + self.n_queue,1)

class AccessTimeOS(ObservationSpace):
    ''' The agent will only see an integer for each spot (0 if empty, 1 if filled with object of type 1 and 2 if filled with object of type 2 etc..) and a real for the access time.

    Notice that we do not give information about the frequency or weight. Since there's a bijection between the object ID and its characteristics we believe it is unnecessary at first.

    This basic OS should allow us to see if the agent can learn to place objects with minimal strategy.
    '''

    def __init__(self, warehouse):
        ''' By convention, the dropping queue is at the end of the observation space!
        '''
        super().__init__(self,warehouse)
        self.space = spaces.Tuple( (spaces.Tuple((spaces.Discrete(n_packages),spaces.Box(0,1))) for _ in range(self.n_spots+self.n_queue)) )

    def compute_state(self):
        return  (  *( (spot.package.id,spot.access_time) if spot.is_full else (0,spot.access_time) for spot in self.spots ), *( (spot.package.id,0) if spot.is_full else (0,0) for spot in self.queue ) )
        
    def input_shape(self):
        return (self.n_spots + self.n_queue,2)