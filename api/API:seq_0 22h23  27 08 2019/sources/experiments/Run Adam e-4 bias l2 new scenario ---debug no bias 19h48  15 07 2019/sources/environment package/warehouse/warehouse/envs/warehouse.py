import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from warehouse.envs.package import Package, generate_packages
from warehouse.envs.spot import Spot
from warehouse.envs.shelve import Shelve

from warehouse.envs.observation_spaces import BasicOS
from warehouse.envs.action_spaces import BasicAS
from warehouse.envs.rewards import BasicReward

from warehouse.envs.utils import flatten, Counter
from warehouse.envs.metrics import compute_storage_efficiency
from warehouse.envs.constants import initial_filled_spots_ratio, energy_cost_fn, packages, plt, min_filling_ratio, max_filling_ratio
from warehouse.envs.rendering import Renderer
from warehouse.envs.callbacks import StatisticsCallback, DescriptionCallback, HistogramCallback

class Warehouse(gym.Env):
    ''' Descibes ONE half of the warehouse structure. A warehouse is comprised of shelve that are comprised of spots that can be filled with packages.

    Each aisle of the warehouse is made up of two unique shelves, one on each side.
    The layers of the warehouse are defined as the individual moving blocks. Each layer is comprised of the shelve that are fixed back to back.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, N=2, render = False, verbose = 0):
        self.seed=123

        self.n_layers = N
        self.max_level = 2*N # the maximum depth of a given shelve
        self.n_shelves = 4*N +1 # counting the center part and the two orientations. (we only claim half of the center shelve)
        

        # Creating the structure in hierarchic way
        self.shelves = [Shelve(0,0,self)] # half of the center shelve
        for level in range(1,self.max_level+1):
            self.shelves += [ Shelve(level,0,self), Shelve(level,1,self) ]

        # Recovering all the spawned spots : the only thing that matters for the training
        self.spots = flatten([shelve.spots for shelve in self.shelves])
        self.n_spots = len(self.spots)
        ## guarantee a simple mapping
        for i in range(self.n_spots): 
         self.spots[i].id = i
        
        # Creating the storing queue
        self.queue = [Spot(0,0,None,coordinates = (-1,1))]
        self.n_queue = len(self.queue)
        
        # Adding metrics
        self.compute_storage_efficiency = lambda : compute_storage_efficiency(self.spots,self.compute_access_cost)
        
        # 1) Fill randomly 50% of the spots with generate_package
        self.init_spawn_packages(ratio=initial_filled_spots_ratio)
        # 2) Add a package on the to store spot
        self.spawn_package()
        
        # Adding some statistics
        self.total_num_spots = len(self.spots)
        self.max_access_time = max([spot.access_time for spot in self.spots])
        self.max_access_cost = self.compute_max_access_cost()
        
        # GYM attributes

        ## The observation space
        self.OS = BasicOS(self)
        self.state = self.OS()

        ## The action space
        self.AS = BasicAS(self)
        self.action_space = self.AS.space

        ## The reward
        self.reward = BasicReward(self)

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

        # Compute the reward
        reward = self.reward(action)

        # Retrieve the description
        description = self.AS.describe(action, verbose=1)

        print('action_descr :', description, 'filling ratio :', self.filling_ratio(),
              'action_reward :', reward)

        # Update the state if needed
        if self.AS.is_drop_package(action) and not self.AS.is_invalid(action):
            target_spot_id = self.AS.drop_to_id(action)
            target_spot = self.spots[target_spot_id]

            reference_spot_id = self.AS.drop_from_id(action)
            reference_spot = self.queue[reference_spot_id]

            moving_package = reference_spot.take()
            target_spot.put(moving_package)


            self.state = self.OS()

        if self.AS.is_move_package(action) and not self.AS.is_invalid(action):
            target_spot_id = self.AS.move_to_id(action)
            target_spot = self.spots[target_spot_id]

            reference_spot_id = self.AS.move_from_id(action)
            reference_spot = self.spots[reference_spot_id]

            moving_package = reference_spot.take()
            target_spot.put(moving_package)

            self.state = self.OS()
            
        
        # Spawn a package if needed
        self.spawn_package()

        # Retrieve a package from the warehouse if there is still more than a given ration of packages
        # if self.filling_ratio() >= min_filling_ratio :
        #     reward += self.retrieve_package()

        if self.filling_ratio() >= max_filling_ratio:
            done=True
            reward += 10


        # Check if the warehouse is full
        if self.is_full():
            reward += 20 # give a very big reward
            done = True
        
        # Verbose option
        if self.verbose > 0:
            print('Step return is : `state`, {}, {}, {}'.format(reward,done,description))
        
        # Return the output
        return [self.state, reward, done, {'str':description}]


    def reset(self):

        # Remove and replace packages on shelves and queue
        self.init_spawn_packages(ratio=initial_filled_spots_ratio, reset=True)
        self.spawn_package(reset=True)

        # Re-compute state
        self.state = self.OS()
        self.done = False

        return self.state

    def render(self, mode='human'):

        if mode == 'human':
        
            _,spot_patches = self.get_patches(shelves = False, spots = True, unrendered_only = True)
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

        if verbose == 1 :
            print("Initial packages spawned, with average access time : {}, and average energy cost : {}".format(*self.compute_storage_efficiency()))


    # Utils


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
        return sum([ energy_cost_fn(max([package[2] for package in packages])*shelve.n_spots) for shelve in self.shelves])
        
    def compute_access_time(self, *spots):
        ''' Compute the time needed to access a sequence of spots.
        '''

        if len(spots) == 0:
            return 0

        if len(spots) == 1:
            return spots[0].access_time

        if len(spots) > 1:
            raise Exception('Not implemented')

    def mask_valid_action(self):



    ## Packages


    def spawn_package(self, reset=False):
        ''' Add a package to be stored if needed.

        This is meant to be called at the end of each epoch and integrates the coniditions for adding a package.
        '''

        if not self.is_empty_queue():
            # We only consider one object to store at a time for now. We can consider a queue later on
            return


        # ADD A HEURISTIC RULE HERE FOR WHEN TO ADD !
        # if np.random.random() > 0.7:
        #     return

        if reset:
            self.reset_storing_queue()

        generated_packages = generate_packages()
        assert len(self.queue) == len(generated_packages) == 1  #Only consider a unique package at the moment
        [spot.put(package) for spot, package in zip(self.queue, generated_packages)]

    def reset_all_spots(self):
        '''
        Remove all packages from spots in all shelves.
        '''
        [spot.take() for spot in self.spots]

    def reset_storing_queue(self):
        '''
        Remove all packages from storing queue.
        '''
        [spot.take() for spot in self.queue if spot.is_full]

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

        candidates = [ (spot.id,spot.access_time) for spot in self.spots if spot.is_full and spot.package.id == package.id ]

        if np.random.random()>0.5:
            return None
        
        if candidates == [] :
            #self.done = True
            return None
        
        candidates.sort( key = lambda t : t[1] )
        
        return self.spots[candidates[0][0]]
        
        
    def retrieve_package(self):
        ''' 
        Retrieve a package from the warehouse and return the associated reward.
        
        This is meant to be called at the end of every step.
        
        TO-DO : add a rule for when to retrieve a package (and how many?)
        '''
        
        if True:
            package = self.sample_package()
            spot = self.choose_spot_from_package(package)
            if spot is None:
                return -1
            _package = spot.take()
            assert _package.id == package.id
            return self.reward.compute_reward_for_retrieval(spot)
            
        return 0

    def filling_ratio(self):
        '''
        Computes the proportion of spots containing a package in the entire warehouse

        '''
        return sum([spot.is_full for spot in self.spots])/len(self.spots)

    def is_full(self):
        for spot in self.spots:
            if not spot.is_full:
                return False
        return True
        

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
                
        if spots:
            for storing_spot in self.queue:
                spot_patch = storing_spot.get_patch(unrendered_only=unrendered_only)
                if spot_patch:
                    spot_patches.append(spot_patch)
                    
        # print('DEBUG : length of shelves patches is {}, length of spots patches is {}'.format(len(shelve_patches,spot_patches)))
         
        return shelve_patches,spot_patches


    # Tests

    def test_energy_cost(self):
        spot = self.shelves[-1].spots[-1]
        cost = self.compute_energy(spot)
        assert cost == 0.0, 'Cost to reach outside cost should be 0, not {}'.format(cost)


