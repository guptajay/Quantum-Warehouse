import numpy as np
import threading
from warehouse.envs.constants.basic_constants import packages


def sum_tuple(l):
    '''
    Computes the tuple-wise sum of a list of tuples.
    Input : [(x_1, y_1), (x_2, y_2], ouput : [(x_1+x_2), (y_1+y_2)]
    '''
    return [sum(x) for x in zip(*l)]

def flatten(l):
    '''
    Flatten a list of list to a single list with every elements
    '''
    return [item for sublist in l for item in sublist]

def mean_tuple(l):
    '''
    Computes the tuple-wise mean of a list of tuples.
    '''
    return [np.array(x).mean() for x in zip(*l)]

def spot_potential_package_ids(size, package):
    spot_packages = package
    spot_size = size
    available_space = spot_size - sum([package.size for package in spot_packages])
    potential_package = []
    for package_id, package_descr in packages.items():
        cur_available_space = available_space
        while package_descr["size"] <= cur_available_space:
            potential_package += [package_id]
            cur_available_space -= package_descr["size"]
    return potential_package

class Counter():

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def __call__(self):
        try:
            self.lock.acquire()
            value = self.value
            self.value += 1
            return value
        except:
            raise Exception('Lock fail')
        finally:
            self.lock.release()

    def reset(self):
        try:
            self.lock.acquire()
            self.value = 0
        except:
            raise Exception('Lock fail')
        finally:
            self.lock.release()

id_counter = Counter()