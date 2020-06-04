import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from warehouse.envs.package import Package, generate_packages
from warehouse.envs.spot import Spot
from warehouse.envs.shelve import Shelve

from warehouse.envs.observation_spaces import BasicOS, AlleyOS
from warehouse.envs.action_spaces import BasicAS, AlleyAS
from warehouse.envs.rewards import BasicReward, RefinedReward

from warehouse.envs.utils import flatten, Counter, spot_potential_package_ids
from warehouse.envs.metrics import compute_storage_efficiency
from warehouse.envs.constants import initial_filled_spots_ratio, energy_cost_fn, packages, plt, min_filling_ratio, \
    max_filling_ratio, spot_size, alley_width, shelve_width, n_packages_per_delivery, n_retrievals_per_day, packages
from warehouse.envs.rendering import Renderer
from warehouse.envs.callbacks import StatisticsCallback, DescriptionCallback, HistogramCallback
from api_utils import APIWarehouseState, APIQueue

import binpacking

import json

class Warehouse(gym.Env):
    ''' Descibes ONE half of the warehouse structure. A warehouse is comprised of shelve that are comprised of spots that can be filled with packages.

    Each aisle of the warehouse is made up of two unique shelves, one on each side.
    The layers of the warehouse are defined as the individual moving blocks. Each layer is comprised of the shelve that are fixed back to back.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, N=2, render = False, verbose = 1, api=False):
        self.seed=123

        self.n_layers = N
        self.max_level = 2*N # the maximum depth of a given shelve
        self.n_shelves = 4*N +1 # counting the center part and the two orientations. (we only claim half of the center shelve)
        

        # Creating the structure in hierarchic way
        self.shelves = [Shelve(0,0,self)] # half of the center shelve
        for level in range(1,self.max_level+1):
            self.shelves += [ Shelve(level,0,self), Shelve(level,1,self) ]

        # Infering the alternative structure : considering alleys instead of shelves
        self.n_alleys = 2*(self.n_layers+1) # we seperate the two orientations
        self.alleys = [ [self.shelves[0],self.shelves[1]], [self.shelves[2]] ] # grouping shelves by alley, the level 0 is with orientation 0
        for i in range(3,self.n_shelves-5,4): # CHECK THIS !!
            self.alleys.append( [self.shelves[i],self.shelves[i+2]] )
            self.alleys.append( [self.shelves[i+1],self.shelves[i+3]] )
        self.alleys.append([self.shelves[-2]])
        self.alleys.append([self.shelves[-1]])


        # Recovering all the spawned spots : the only thing that matters for the training
        self.spots = flatten([shelve.spots for shelve in self.shelves])
        self.n_spots = len(self.spots)
        ## guarantee a simple mapping
        for i in range(self.n_spots): 
            self.spots[i].id = i
        
        # Creating the storing queue
        self.queue = [Spot(0,0,None,coordinates = (-1,1),size=-1)]
        self.n_queue = len(self.queue)
        
        # Adding metrics
        self.compute_storage_efficiency = lambda : compute_storage_efficiency(self.spots,self.compute_access_cost)

        # Init API
        self.api=api
        self.warehouse_api = APIWarehouseState(fname="warehouse")
        self.queue_api = APIQueue(fname='queue')
        
        # 1) Fill randomly 50% of the spots with generate_package
        self.init_spawn_packages(ratio=initial_filled_spots_ratio, api=self.api)
        # 2) Add a package on the to store spot
        self.spawn_package(n_packages_per_delivery(), api=self.api, init=True)
        
        # Adding some statistics
        self.total_num_spots = len(self.spots)
        self.max_access_time = max([spot.access_time for spot in self.spots])
        self.max_access_cost = self.compute_max_access_cost()
        
        # GYM attributes

        ## The observation space
        self.OS = AlleyOS(self)
        self.state = self.OS()

        ## The action space

        ### This is complexified in our "simple version". We want the agent to choose an alley. 
        ### We then compute the best spot within this package. Then we compute the reward from picking
        ### that spot and give that to the agent. Problem is : the reward requires the AS to call
        ### its methods "is_drop_package", "drop_from_id" etc...
        ### For easier adaptation, we hence keep two seperate Action Spaces. An alley one that will be used
        ### by gym to predict an action and another one for the reward class to predict the reward.
        ### Rewards will look for warehouse.AS; let's not change that.

        self.AS = BasicAS(self)
        self.gym_AS = AlleyAS(self)
        
        self.action_space = self.gym_AS.space


        ## The reward
        # self.reward = BasicReward(self)
        self.reward = RefinedReward(self)

        # Game completion
        self.done = False

        # Rendering
        fig_shape = ( (-self.n_layers-1, self.n_layers+2), (-self.n_layers-1, self.n_layers+1) )
        shelve_patches, spot_patches = self.get_patches()
        self.current_stats = { 'current_action' : '', 'current_episode':0, 'current_reward':0, 'current_step':0}
        self.viewer = Renderer(fig_shape,shelve_patches,spot_patches,stats=self.current_stats)
        self.verbose = verbose
        
        # Callbacks
        self.statistics_callback = StatisticsCallback(self)
        self.description_callback = DescriptionCallback(self)
        self.histogram_callback = HistogramCallback()
        self.callbacks = [self.statistics_callback, self.description_callback, self.histogram_callback]

        # Real-life options
        ## We want to retrieve objects during 7 days and put new objects during 1 day
        ## The easiest way to implement this is to keep everything as is but only 
        ## retrieve objects once every few steps. "Every few" = the number of packages
        ## we want to drop on the 7th day and when we retrieve packages we retrieve all 
        ## at once all those from the 6 remaining days. This way, at every step, the 
        ## agent still needs to take an action. We define the constants relating to this
        ## in the real_life_constants.py


    '''
    Core functions
    '''

    def step(self, action):
        '''
        Simplified step function when only predicting destination spot
        # Arg
         - action : Discrete : Destination spot
         By convention, -1 for do-nothing action
        # Output
         -
        '''
        assert self.action_space.contains(action), "{} ({}) is not contained in the action space ({})".format(action, type(action),self.AS)

        # If game completed, return current state
        if self.done :
            print("\nGame Over")
            return [self.state, 0, self.done, {'str':'learning completed'}]

        # A priory, done is false
        done = False

        alley_action = action # store the alley action

        # Compute the spot-action from the alley-action
        if self.gym_AS.is_drop_package(alley_action):
            reference_spot_id = self.AS.drop_from_id(alley_action)
            reference_spot = self.queue[reference_spot_id]
            moving_package = reference_spot.package[0]
            action = self.alley_to_spot_action(alley_action,moving_package)

            if action is None: #Occur with random steps
                return [self.state, 0, done, {'str':'', 'mask':json.dumps([1 if self.available_spot_in_alley(alley, self.queue[0].package[0]) else 0 for alley in self.alleys])}]


        retrieval_factor =  -1
        dropping_factor = -2
        batch_retrieval_factor = -.1
        end_episode_factor = 0

        # Compute the reward
        reward = self.reward(action)*dropping_factor

        # Retrieve the description
        description = self.AS.describe(action, verbose=1)

        # print(description, 'filling ratio :', self.filling_ratio(),
        #       'action_reward :', reward)

        # Update the state if needed
        if self.AS.is_drop_package(action) and not self.AS.is_invalid(action):
            target_spot_id = self.AS.drop_to_id(action)
            target_spot = self.spots[target_spot_id]

            reference_spot_id = self.AS.drop_from_id(action)
            reference_spot = self.queue[reference_spot_id]

            moving_package = reference_spot.take(reference_spot.package[0].id)
            target_spot.put(moving_package)



        if self.AS.is_move_package(action) and not self.AS.is_invalid(action):
            raise NotImplementedError
            # target_spot_id = self.AS.move_to_id(action)
            # target_spot = self.spots[target_spot_id]
            #
            # reference_spot_id = self.AS.move_from_id(action)
            # reference_spot = self.spots[reference_spot_id]
            #
            # moving_package = reference_spot.take()
            # target_spot.put(moving_package)
            #
            # self.state = self.OS()

        if self.queue[0].is_empty():

            # We finished placing all the packages from the last delivery.
            ## Retrieve for 6 days and accept a new delivery.

            for day in range(6):
                for _ in range(n_retrievals_per_day()):
                    if all([spot.is_empty() for spot in self.spots]):
                        done = True
                        print('Warehouse empty, no package to retrieve')
                        break
                    else:
                        reward += self.retrieve_package()*retrieval_factor

            self.spawn_package(n_packages_per_delivery(), api=self.api)

            if self.verbose>0:
                print('Empty queue: retrieving packages with cumulative reward : {}'.format(reward))


        mask = [1 if self.available_spot_in_alley(alley, self.queue[0].package[0]) else 0 for alley in self.alleys]

        self.state = self.OS()

        # Check if the warehouse is full
        if done or self.filling_ratio() >= max_filling_ratio or self.is_full() or sum(mask)==0:
            reward += self.reward.episode_completion_reward()*end_episode_factor
            done = True

        
        if batch_retrieval_factor > 0:
            reward += self.reward.compute_reward_for_batch_retrieval()*batch_retrieval_factor

        # Verbose option
        if self.verbose > 0:
            print('Step return is : `state`, {}, {}, {}'.format(reward,done,description))

        # Update API
        if self.api:
            spot_ids = []
            warehouse_package_ids = []
            for spot in self.spots:
                spot_ids.append(spot.id)
                spot_packages = [package.id for package in spot.package]
                padded_spot_package = spot_packages + [0, ]*(spot_size - len(spot_packages))
                warehouse_package_ids.append(padded_spot_package)


            self.warehouse_api.to_csv(spot_ids=spot_ids, packages_ids=warehouse_package_ids)
            queue_package_ids = [package.id for package in self.queue[0].package]
            self.queue_api.to_csv(package_ids=queue_package_ids)

        # Return the output
        return [self.state, reward, done, {'str':description, 'mask':json.dumps(mask)}]


    def reset(self):

        # Reset renderer
        if self.viewer is not None:
            self.viewer.reset()

        # Remove and replace packages on shelves and queue
        self.init_spawn_packages(ratio=initial_filled_spots_ratio(), reset=True, api=self.api)
        self.spawn_package(n_packages_per_delivery(), reset=True, api=self.api)

        # Re-compute state
        self.state = self.OS()
        self.done = False

        return self.state

    def render(self, mode='human'):

        if mode == 'human':
        
            _,spot_patches = self.get_patches(shelves = False, spots = True, unrendered_only = False)
            self.viewer.update(spot_patches)

        else:
            raise Exception('Unrecognized rendering mode : {}'.format(mode))

    def describe(self, cascade = True):
        print('Warehouse with {} layers containing {} spots in {} shelves, and a max level of {}'.format(self.n_layers, self.total_num_spots, self.n_shelves, self.max_level))
        print('Describing my shelves : ')
        if cascade:
            for shelve in self.shelves:
                shelve.describe('   ')


    '''
    Support functions
    '''

    ## Init

    def init_spawn_packages(self, ratio=.5, reset=False, verbose=1, api=False):
        '''
        Fills randomly a part of the entire storage spots
        '''
        if reset:
            self.reset_all_spots()

        if not api:
            n_packages = int(self.n_spots*ratio)
            packages_ = generate_packages(number=n_packages)

            ## Uniformely draw the position on the list of spots the position to place the packages
            # Retrieve all spots and merge them to a unique list

            # Sample spots without replace
            assert len(self.spots) >= n_packages, 'Inconsistant number of spots to fill at init'
            spot_choices = np.random.choice(self.spots, n_packages, replace=False)

            # Fill spots
            assert len(spot_choices)==len(packages_), 'Inconsistancy between sampled spots and sampled package'
            [spot.put(package) for (spot, package) in zip(spot_choices, packages_)]

            if verbose == 1 :
                print("Initial packages spawned, with average access time : {}, and average energy cost : {}".format(*self.compute_storage_efficiency()))
        else:
            #Spawn package based on the
            api_state = self.warehouse_api.from_csv(init=True)
            spot_ids = np.array(api_state["spot_ids"])
            packages_ids = np.array(api_state["package_ids"])
            mask, *_ = np.where(np.sum(packages_ids, axis=1)>0) #TODO: the sum trick only because init can only put
            # one package per spot
            [self.retrieve_spot_from_id(spot_id).put(Package(package_id, *list(packages[package_id].values()))) for (
                spot_id,
                                                                                                            package_id) in zip(spot_ids[mask],
                                                                                                 packages_ids[mask][
                                                                                                 :, 0])]


    # Utils

    def retrieve_spot_from_id(self, spot_id):
        for spot in self.spots:
            if spot.id == spot_id:
                return spot
        return None

    ## Costs

    def compute_access_cost(self, *spots): 
        '''
        Compute the dollar equivalent energy required to reach a sequence of spots.

        #Args
            - *spots : sequence of spots 
        '''

        if len(spots)==1:
            spot = spots[0]
            level = spot.shelve.level
            orientation = spot.shelve.orientation
            ge = lambda x, l: x > l if l%2==0 else x >= l

            # Select shelve outside the current shelve of the spot
            outside_shelves = [shelve for shelve in self.shelves if (shelve.orientation == orientation and ge(shelve.level, level))]
            # Compute resulting energy
            energy = np.array([shelve.moving_energy_cost() for shelve in outside_shelves]).sum()

            return energy
        
        else:
            raise Exception('Sequence of spots not implemented')

    def compute_max_access_cost(self): 
        '''
        Compute the maximum dollar equivalent energy required to reach a sequence of spots.
        '''
        return sum([ energy_cost_fn(max([packages[package_id]['weight'] for package_id in packages])*shelve.n_spots) for shelve in self.shelves])
        
    def compute_access_time(self, *spots):
        ''' Compute the time needed to access a sequence of spots.
        '''

        if len(spots) == 0:
            return 0

        if len(spots) == 1:
            return spots[0].access_time

        if len(spots) > 1:
            raise Exception('Not implemented')

    ## Packages


    def spawn_package(self, n, reset=False, api=False, init=False):
        ''' Add a package to be stored if needed.

        This is meant to be called at the end of each epoch and integrates the coniditions for adding a package.
        '''

        if reset:
            self.reset_storing_queue()

        if not init or not api:
            for package in generate_packages(n):
                self.queue[0].put(package)

        else:
            queue_api = self.queue_api.from_csv(init=True)
            for index in queue_api["package_ids"]:
                self.queue[0].put(Package(index,*list(packages[index].values())))


    def reset_all_spots(self):
        '''
        Remove all packages from spots in all shelves.
        '''
        [spot.empty() for spot in self.spots]

    def reset_storing_queue(self):
        '''
        Remove all packages from storing queue.
        '''
        [[spot.take(package.id) for package in spot.package] for spot in self.queue if len(spot.package)>0]

    def is_empty_queue(self):
        '''
        Return True if the dropping queue is empty.
        '''
        return not np.array([spot.is_full for spot in self.queue]).any()

    def sample_package(self):
        ''' 
        Sample a package following the frequency distribution.
        '''
        return generate_packages(1)[0]
        
    def choose_spot_from_package(self,package):
        '''
        Given a package, choose the spot from the warehouse that will be used to get this package.
        
        Construction Constraint : such a spot should always exist! Otherwise we stop the training.
        '''

        candidates = [(spot.id,spot.access_time) for spot in self.spots if not spot.is_empty() and
                       any([package_.id == package.id for package_ in spot.package])]

        if candidates == [] :
            return None
        
        candidates.sort( key = lambda t : t[1] )
        
        return self.spots[candidates[0][0]]
        
        
    def retrieve_package(self, reward=False):
        ''' 
        Retrieve a package from the warehouse and return the associated reward.
        
        This is meant to be called at the end of every step.
        '''

        package = self.sample_package()
        spot = self.choose_spot_from_package(package)   
        while spot is None:
            package = self.sample_package()
            spot = self.choose_spot_from_package(package)
        _package = spot.take(package.id) if not reward else spot.package
        assert _package.id == package.id
        reward = self.reward.compute_reward_for_retrieval(spot)
        return reward

    def filling_ratio(self):
        '''
        Computes the proportion of spots containing a package in the entire warehouse

        '''
        return sum([len(spot.package) for spot in self.spots])/(len(self.spots)*spot_size)

    def empty_spot_ratio(self):
        return sum([spot.is_empty for spot in self.spots])/len(self.spots)

    def is_full(self):
        for spot in self.spots:
            if not spot.is_full:
                return False
        return True

    def current_env_packages(self):
        packages = [spot.package for spot in self.spots if spot.is_full]
        return packages

    @staticmethod
    def available_spot_in_alley(alley, package):
        id = package.id
        for shelve in alley: #TODO: change function to iterwhile
            for spot in shelve.spots:
                if id in spot_potential_package_ids(spot.size, spot.package):
                    return True
        return False

    ## Rendering utils
    
    def get_patches(self, shelves = True, spots = True, unrendered_only = True):
        
        if not shelves and not spots:
            return
            
        shelve_patches, spot_patches = [] if shelves else None, [] if spots else None
        for shelve in self.shelves:
            _shelves_patch, _spot_patches = shelve.get_patches(spots = True, unrendered_only = unrendered_only)
            if shelves:
                shelve_patches.append(_shelves_patch)
            if spots:
                spot_patches += _spot_patches
                
        if spots:
            for storing_spot in self.queue:
                spot_patch = storing_spot.get_patch(unrendered_only=unrendered_only,queue=True)

                if spot_patch:
                    if isinstance(spot_patch,list):
                        spot_patches += spot_patch
                    else:
                        spot_patches.append(spot_patch)
                    
        # print('DEBUG : length of shelves patches is {}, length of spots patches is {}'.format(len(shelve_patches,spot_patches)))
         
        return shelve_patches,spot_patches

    # Binpacking

    @staticmethod
    def to_spot_volume(b):
        bins = binpacking.to_constant_volume(b, spot_size)
        return bins

    def minimum_spot_given_current_packages(self):
        '''
        Returns the minimum number of spots given the packages in the warehouse
        '''
        pass

    def packages_to_bin_dict(self):
        '''
        Transform the list of packages to a dictionnary to be processed by binpacking library.
        Input = [Packages(f:.., w:.., s:..), .. ], output {'id': size, .. }
        '''
        pass

    def additional_boxes_to_place(self, remaining_spots):
        '''
        Computes the number of possible additional boxes given the number of remaining available spots and the respective frequency of the packages
        '''
        pass




    # Alley action space functions

    def alley_to_spot_action(self,alley_index,package):
        
        alley = self.alleys[alley_index]

        possible_spots = flatten([ [spot for spot in shelve.spots if package.id in spot_potential_package_ids(spot.size,spot.package)] for shelve in alley])

        possible_spots.sort(key = lambda spot : spot.access_time)

        try :
            return possible_spots[0].id
        except IndexError:
            print("Impossible to place package, probably occurred during random actions")
            return None


from warehouse.envs.shelve import ParallelShelve

class ParallelWarehouse(gym.Env):
    ''' Descibes a classic warehouse for comparing with ours.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, n_shelves=10, n_spots_per_row = 10, render = False, verbose = 1):
        self.seed=123

        self.n_shelves = n_shelves
        self.n_spots_per_shelve = n_spots_per_row

        # Creating the structure in hierarchic way
        self.shelves = [ ParallelShelve(i,n_spots = self.n_spots_per_shelve, warehouse = self) for i in range(self.n_shelves)]

        # Recovering all the spawned spots : the only thing that matters for the training
        self.spots = flatten([shelve.spots for shelve in self.shelves])
        self.n_spots = len(self.spots)
        ## guarantee a simple mapping
        for i in range(self.n_spots): 
            self.spots[i].id = i
        
        # Rendering
        fig_shape = ( (-1,int(np.ceil(self.n_shelves/2))*alley_width + self.n_shelves*shelve_width +1), (-1,self.n_spots_per_shelve/2 + 1) )
        shelve_patches, spot_patches = self.get_patches()
        self.current_stats = {}
        # TO SAVE IMAGES : set the save_path to an existing directory.
        self.viewer = Renderer(fig_shape,shelve_patches,spot_patches,stats=self.current_stats,save_path=None)
        self.verbose = verbose
        
    '''
    Core functions
    '''

    def reset(self):

        # Reset renderer
        if self.viewer is not None:
            self.viewer.reset()

        # Remove and replace packages on shelves and queue
        self.init_spawn_packages(ratio=initial_filled_spots_ratio(), reset=True, api=self.api)

    def render(self, mode='human'):

        if mode == 'human':
        
            _,spot_patches = self.get_patches(shelves = False, spots = True, unrendered_only = False)
            self.viewer.update(spot_patches)

        else:
            raise Exception('Unrecognized rendering mode : {}'.format(mode))

    '''
    Support functions
    '''

    ## Init

    def init_spawn_packages(self, ratio=.5, reset=False, verbose=1):
        '''
        Fills randomly a part of the entire storage spots
        '''

        n_packages = int(self.n_spots*ratio)
        packages = generate_packages(number=n_packages)

        if reset:
            self.reset_all_spots()

        ## Uniformely draw the position on the list of spots the position to place the packages
        # Retrieve all spots and merge them to a unique list

        # Sample spots without replace
        assert len(self.spots) >= n_packages, 'Inconsistant number of spots to fill at init'
        spot_choices = np.random.choice(self.spots, n_packages, replace=False)

        # Fill spots
        assert len(spot_choices)==len(packages), 'Inconsistancy between sampled spots and sampled package'
        [spot.put(package) for (spot, package) in zip(spot_choices, packages)]


    def compute_access_cost(self, *spots): 
        '''
        Compute the dollar equivalent energy required to reach a sequence of spots.

        #Args
            - *spots : sequence of spots 
        '''

        if len(spots)==1:
            spot = spots[0]
            level = spot.shelve.level
            orientation = spot.shelve.orientation
            ge = lambda x, l: x > l if l%2==0 else x >= l

            # Select shelve outside the current shelve of the spot
            outside_shelves = [shelve for shelve in self.shelves if (shelve.orientation == orientation and ge(shelve.level, level))]
            # Compute resulting energy
            energy = np.array([shelve.moving_energy_cost() for shelve in outside_shelves]).sum()

            return energy
        
        else:
            raise Exception('Sequence of spots not implemented')

    def compute_max_access_cost(self): 
        '''
        Compute the maximum dollar equivalent energy required to reach a sequence of spots.
        '''
        return sum([ energy_cost_fn(max([packages[package_id]['weight'] for package_id in packages])*shelve.n_spots) for shelve in self.shelves])
        
    def compute_access_time(self, *spots):
        ''' Compute the time needed to access a sequence of spots.
        '''

        if len(spots) == 0:
            return 0

        if len(spots) == 1:
            return spots[0].access_time

        if len(spots) > 1:
            raise Exception('Not implemented')

    ## Packages


    def reset_all_spots(self):
        '''
        Remove all packages from spots in all shelves.
        '''
        [spot.empty() for spot in self.spots]


    def sample_package(self):
        ''' 
        Sample a package following the frequency distribution.
        '''
        return generate_packages(1)[0]
        
    def choose_spot_from_package(self,package):
        '''
        Given a package, choose the spot from the warehouse that will be used to get this package.
        
        Construction Constraint : such a spot should always exist! Otherwise we stop the training.
        '''

        candidates = [ (spot.id,spot.access_time) for spot in self.spots if spot.is_full and spot.package.id == package.id ]

        if candidates == [] :
            return None
        
        candidates.sort( key = lambda t : t[1] )
        
        return self.spots[candidates[0][0]]


    def filling_ratio(self):
        '''
        Computes the proportion of spots containing a package in the entire warehouse

        '''
        return sum([spot.is_full for spot in self.spots])/len(self.spots)

    def empty_spot_ratio(self):
        return sum([spot.is_empty for spot in self.spots])/len(self.spots)

    def is_full(self):
        for spot in self.spots:
            if not spot.is_full:
                return False
        return True

    def current_env_packages(self):
        packages = [spot.package for spot in self.spots if spot.is_full]
        return packages

    ## Rendering utils
    
    def get_patches(self, shelves = True, spots = True, unrendered_only = False):
        
        if not shelves and not spots:
            return
            
        shelve_patches, spot_patches = [] if shelves else None, [] if spots else None
        for shelve in self.shelves:
            _shelves_patch, _spot_patches = shelve.get_patches(spots = True, unrendered_only = unrendered_only)
            if shelves:
                shelve_patches.append(_shelves_patch)
            if spots:
                spot_patches += _spot_patches
                    
        return shelve_patches,spot_patches