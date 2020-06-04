import numpy as np

# These should be made more carefully! 
# n_packages_per_delivery = lambda : np.random.randint(47,51)
# n_retrievals_per_day = lambda : np.random.randint(7,8)

n_packages_per_delivery = lambda : np.random.randint(5,15)
n_retrievals_per_day = lambda : np.random.randint(1,3)