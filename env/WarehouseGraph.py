import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors, style

GRID_SIZE = 7


class WarehouseGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, warehouse_state):

        self.warehouse_state = warehouse_state

        data = np.random.rand(GRID_SIZE, GRID_SIZE) * 20

        k = 0

        # TODO - This is incorrect, need to fix
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                data[i][j] = warehouse_state[k][1]
                k = k + 1

        # create discrete colormap
        cmap = colors.ListedColormap(['blue', 'red'])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='major', axis='both',
                linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, 7, 1))
        ax.set_yticks(np.arange(-.5, 7, 1))
        plt.suptitle("Quantum Warehouse")
        plt.show(block=False)
        plt.pause(5)
        self.close()

    def close(self):
        plt.close()
