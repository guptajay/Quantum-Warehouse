import numpy as np
from warehouse.envs.constants import Constants

class Step(object):

    def __init__(self, name):
        self.name = name
        self.func = globals()[name]


    def step_baseline(self, warehouse, action):
        '''
        Simplified step function when only predicting destination spot
        # Arg
         - action : Discrete : Destination spot
         By convention, -1 for do-nothing action
        # Output
         -
        '''
        assert warehouse.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # If game completed, return current state
        if warehouse.done:  # or self.counter==self.max_counter:  Directly defined in keras settings nb_max_episode_steps
            print("Learning completed")
            warehouse.done = True
            return [warehouse.state, 0, warehouse.done, {'str': 'learning completed'}]

        # If do-nothing : return current state
        elif action == -1:
            reward = -1
            return [warehouse.state, reward, False, {'str': 'do-nothing'}]

        # If predicts to move a box to the package to store location
        elif action == 0:
            reward = -1
            return [warehouse.state, reward, False, {'str': 'trying to put box to the storing location'}]

        # If action impossible (destination box filled), return current state
        elif warehouse.id_spot_mapping[action].is_full:
            # print('Trying to place a package to a filled spot')
            reward = -1
            return [warehouse.state, reward, False, {'str': 'Incorrect spot: already filled'}]

        # Else, place the good at the given destination, compute reward, remove package from storing queue
        else:
            target_spot = warehouse.id_spot_mapping[action]

            # Fill spot with package from queue
            target_spot.put(
                warehouse.storing_queue[0].package)  # TODO: check that the spot is update on the shelve object

            # Remove package from queue
            warehouse.storing_queue[0].take()

            # Update state
            all_spots = warehouse.retrieve_all_spots()
            # Retreive full spots
            filled_spots = [spot for spot in all_spots if (spot.is_full and not warehouse.spot_id_mapping[spot] == 0)]
            # If storage reach a certain level of filling, remove randomly certain spots
            if len(filled_spots) >= len(all_spots)*Constants.max_load_factor:
                spot_to_empty = np.random.choice(filled_spots)
                spot_to_empty.take()
            # Compute resulting state
            warehouse.compute_state(all_spots)

            # Append new package to queue and increment counter
            warehouse.spawn_package()

            # Compute reward
            reward = 2 - target_spot.access_time/warehouse.max_access_time - warehouse.compute_energy_to_access(
                target_spot)*1000  # TODO: think of relevant way to scale energy cost, scaling by max energy cost would require to recompute an entire warehouse config.
            return [warehouse.state, reward, False, {'str': 'Moved box from {} to {}'.format(0, action)}]
