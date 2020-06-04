import numpy as np
import threading

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