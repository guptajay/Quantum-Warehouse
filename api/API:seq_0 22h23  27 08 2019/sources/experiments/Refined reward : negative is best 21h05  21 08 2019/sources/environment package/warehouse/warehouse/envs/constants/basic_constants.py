import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.colors import Normalize

# Distances
unit_distance = 2 #
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

# Spot
spot_size = 1

# Packages
packages = {
    # ( id, frequency, weight, size)
#     1: {
#         'frequency' :0.8,
#         'weight':0.1,
#         'size' :1
#     },
#     2: {
#         'frequency' :0.17,
#         'weight':0.4,
#         'size' :2
#     },
#     3: {
#         'frequency' :0.03,
#         'weight':100,
#         'size' :4
#     }
# }
#     1: {
#         'frequency' :0.3,
#         'weight':0.1,
#         'size' :1
#     },
#     2: {
#         'frequency' :0.05,
#         'weight':0.4,
#         'size' :2
#     },
#     3: {
#         'frequency' :0.15,
#         'weight':0.8,
#         'size' :4
#     },
#     4: {
#         'frequency': 0.2,
#         'weight': 0.8,
#         'size': 1
#     },
#     5: {
#         'frequency': 0.05,
#         'weight': 0.1,
#         'size': 2
#     },
#     6: {
#         'frequency': 0.25,
#         'weight': 2,
#         'size': 1
#     }
#
# }
# {
    1: {
        'frequency' :0.8,
        'weight':0.1,
        'size' :1
    },
    2: {
        'frequency' :0.2,
        'weight':0.4,
        'size' :1
    }
}
n_packages = 2
max_weight = max([package["weight"] for package in packages.values()])

# Initial configuration
initial_filled_spots_ratio = 0.5

# Maximum load of warehouse
max_filling_ratio = .93
min_filling_ratio = .5

# Frequecy
min_frequency = min(packages[package_id]['frequency'] for package_id in packages)
max_frequency = max(packages[package_id]['frequency'] for package_id in packages)

# Color Maps
# frequency_cmap = lambda value : plt.cm.Blues(Normalize(vmin=0, vmax=max_frequency)(value))
frequency_cmap = lambda value : plt.cm.RdYlBu(Normalize(vmin=0, vmax=max_frequency)(value))