import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.colors import Normalize

# Distances
unit_distance = 1
unit_diagonal_distance = np.sqrt(2*(unit_distance ** 2))

# Times
unit_opening_time = 10
unit_drive_time = 1
unit_diagonal_drive_time = np.sqrt(2*(unit_drive_time ** 2))

# Energy
gravitational_pull = 9.81
electricity_cost = 6.625e-8
energy_cost_fn = lambda weight : weight*gravitational_pull*electricity_cost

# Shelves
unit_spots_per_block = 1 #takes into account the height as well

# Packages
packages = [
    # ( id, frequency, weight )
    (0, 0.7, 0.1),
    (1, 0.3, 0.4)
]
n_packages = 2

# Initial configuration
initial_fileld_spots_ratio = .5

# Maximum load of warehouse
max_load_factor = .5

# Frequecy
min_frequency = min( package[1] for package in packages)
max_frequency = max( package[1] for package in packages)

# Color Maps
frequency_cmap = lambda value : plt.cm.Blues(Normalize(vmin=0, vmax=max_frequency)(value))