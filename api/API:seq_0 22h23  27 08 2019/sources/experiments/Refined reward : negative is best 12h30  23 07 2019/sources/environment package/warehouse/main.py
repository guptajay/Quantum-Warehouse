from warehouse.envs import *
import numpy as np
import gym

import os
import matplotlib.pyplot as plt
plt.ion()

def test_init(N = 1):

    warehouse = Warehouse(N)

    # warehouse.describe()

    warehouse.render()

    count = 0

    while count <100 and not warehouse.done:

        count += 1
        # action = (0, np.random.randint(0,60, 1)[0])
        # action = tuple(np.random.randint(0,60, 2))
        action = np.random.randint(0, 60)
        new_state, reward, done, info = warehouse.step(action)
        print("Reward : {} for action : {}".format(reward, info['str']))
        warehouse.render()

if __name__ == "__main__":

    print('Running test init')
    test_init(3)

    
    
    



