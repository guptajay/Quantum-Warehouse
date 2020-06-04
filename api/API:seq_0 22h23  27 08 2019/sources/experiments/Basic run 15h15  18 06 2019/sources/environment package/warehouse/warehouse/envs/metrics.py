import numpy as np

from warehouse.envs.utils import flatten,mean_tuple

def compute_storage_efficiency(spots, energy_fn, n=100):
    '''
    Simple function to compute the time-energy efficiency of the current state.
    Represents the average cost and energy to reach the spots.
    Computed by sampling spots based on the frequency of their parckage, with replacement.
    '''
    # Retreive spots of given state

    # Compute frequency of appearance of each package stored
    filled_spots_frequency = [spot.package.frequency if spot.is_full else 0 for spot in spots]
    normalized_frequency = np.array(filled_spots_frequency)/sum(filled_spots_frequency)

    # Sample packages to retrieve
    indexes_to_retreive = np.random.multinomial(n, normalized_frequency)

    # Compute cost as (average access time, average energy cost)
    spot_cost = lambda spot: (spot.access_time, energy_fn(spot))
    spot_costs = flatten([[spot_cost(spot)]*i for spot, i in zip(spots, indexes_to_retreive)])
    mean_spot_costs = mean_tuple(spot_costs)

    return mean_spot_costs