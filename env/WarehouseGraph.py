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

        # Warehouse Mapping
        # Top Quarter
        k = 0
        for j in range(7):
            data[0][j] = warehouse_state[k][1]
            k = k + 1

        k = 24
        for j in range(1, 6):
            data[1][j] = warehouse_state[k][1]
            k = k + 1

        k = 40
        for j in range(2, 5):
            data[2][j] = warehouse_state[k][1]
            k = k + 1

        # Bottom Quarter
        k = 12
        for j in range(6, -1, -1):
            data[6][j] = warehouse_state[k][1]
            k = k + 1

        k = 32
        for j in range(5, 0, -1):
            data[5][j] = warehouse_state[k][1]
            k = k + 1

        k = 44
        for j in range(4, 1, -1):
            data[4][j] = warehouse_state[k][1]
            k = k + 1

        # Left Quarter
        k = 19
        for j in range(5, 0, -1):
            data[j][0] = warehouse_state[k][1]
            k = k + 1

        k = 37
        for j in range(4, 1, -1):
            data[j][1] = warehouse_state[k][1]
            k = k + 1

        data[3][2] = warehouse_state[47][1]

        # Right Quarter
        k = 7
        for j in range(1, 6):
            data[j][6] = warehouse_state[k][1]
            k = k + 1

        k = 29
        for j in range(2, 5):
            data[j][5] = warehouse_state[k][1]
            k = k + 1

        data[3][4] = warehouse_state[43][1]

        # Center
        data[3][3] = warehouse_state[48][1]

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
        plt.pause(2)
        self.close()

    def close(self):
        plt.close()
