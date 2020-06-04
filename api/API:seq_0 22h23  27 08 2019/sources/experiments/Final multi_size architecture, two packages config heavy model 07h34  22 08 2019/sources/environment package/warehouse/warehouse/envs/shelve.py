from matplotlib.patches import Rectangle
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from warehouse.envs.spot import Spot
from warehouse.envs.constants import unit_spots_per_block, unit_opening_time, unit_diagonal_drive_time, \
    unit_drive_time, energy_cost_fn, spot_size, alley_width, shelve_width


class Shelve():
    '''
    A shelve contains a single array of spots.

    The center shelve has a different behavior and is treated separately.

    The access energy is the same for all spots within a shelve but the access time is not. On the hand, the access energy depends on the current state of the warehouse (the weights on the shelves that need to be moved influences a lot) while the access time is is constant.
    
    # Args
     - level : the depth of the shelve where 0 is the center and N is the outer shelve
     - orientation : 0 or 1. To differentiate the two possible orientations, 0 being the one that holds the corner.
     - warehouse : the warehouse to which this shelve belongs
    '''

    def __init__(self, level, orientation,warehouse):

        assert level!=0 or orientation==0, 'We cannot claim the center shelve with orientation 1'

        self.level = level
        self.orientation = orientation
        
        # Initialising spots
        self.n_spots = (2 if level == 0 else 2*2*(int(np.ceil(level/2))))*unit_spots_per_block
        self.spots = []
        
        # Coordinates
        center_point   = np.array(((1-self.orientation)/2,0))
        self.bottom_left_corner = center_point + ( np.array( (0.5*self.level,-int(np.ceil(self.level/2))) ) if self.orientation == 0 else np.array( (-int(np.ceil(self.level/2)),-0.5*self.level  )  )  )
        shelve_width = 1/2 if self.orientation == 0 else self.n_spots/2
        shelve_height = 1/2 if self.orientation == 1 else self.n_spots/2
        
        # Creating spots (they need the coordinates)
        for spot in range(1,self.n_spots+1):
            # see notes.md for details
            acccess_time = unit_opening_time * ( 1 if level < warehouse.max_level else 0 ) + unit_diagonal_drive_time * (warehouse.max_level-level) + ( spot + (-1 if self.orientation == 0 else 1)*(1 if level%2==(1 if self.orientation == 0 else 0) else 0) ) * unit_drive_time/2
            self.spots.append(Spot(acccess_time, spot if self.orientation == 1 else self.n_spots - spot + 1, shelve = self, size=spot_size))

        # Internal states
        self.warehouse = warehouse
        
        # Rendering box
        self.shelve_box = Rectangle( self.bottom_left_corner, shelve_width, shelve_height ,edgecolor='k',facecolor='none')


    def weight(self):
        ''' Compute the cumulative weight of the shelve
        '''
        return sum([spot.package.weight if spot.is_full else 0 for spot in self.spots])

    def moving_energy_cost(self):
        '''
        Compute cost in dollar for moving the aisle given its weight and physical properties - used to compute reward
        :return: Float -
        '''
        return energy_cost_fn(self.weight())

    def __str__(self):
        self.describe()


    def describe(self, prefix = '', cascade = True):
        ''' Print a verbose description of the shelve
        '''
        print(prefix,'Shelve of orientation {}, level {} with {} spots'.format(self.orientation,self.level,self.n_spots))
        print(prefix,'My spots are : ')
        if cascade:
            for spot in self.spots:
                spot.describe(prefix+'   ')


    def plot(self, ax = None, verbose = 0):
        ''' Plot the shelve on an ax using matplotlib rectangles.

        If an ax is given, the rectangle will be added to it and the plot will not be rendered.
        '''
        render = False

        if ax is None:
            render = True
            fig, ax = plt.subplots(1)
            ax.set_title('Shelve')
            
        ax.add_patch(self.shelve_box)

        if verbose > 0:
            for spot in self.spots:
                spot.plot(ax,self.bottom_left_corner,verbose=verbose-1)

        if render:
            plt.show()
            
    def get_patch(self):
        return self.shelve_box
    
    def get_patches(self,spots=True, unrendered_only=False):
        
        if spots :
            spot_patches = []
            for spot in self.spots:
                spot_patch = spot.get_patch(unrendered_only=unrendered_only)
                if spot_patch:
                    if isinstance(spot_patch,list):
                        spot_patches += spot_patch
                    else:
                        spot_patches.append(spot_patch)
            return self.get_patch(),spot_patches
        else:
            return self.get_patch(),None
            



class ParallelShelve():
    '''
    A shelve contains a single array of spots.

    The center shelve has a different behavior and is treated separately.

    The access energy is the same for all spots within a shelve but the access time is not. On the hand, the access energy depends on the current state of the warehouse (the weights on the shelves that need to be moved influences a lot) while the access time is is constant.
    
    # Args
     - level : the depth of the shelve where 0 is the center and N is the outer shelve
     - orientation : 0 or 1. To differentiate the two possible orientations, 0 being the one that holds the corner.
     - warehouse : the warehouse to which this shelve belongs
    '''

    def __init__(self, level, n_spots, warehouse):

        self.level = level
        
        # Initialising spots
        self.n_spots = n_spots
        self.spots = []

        
        # Internal states
        self.warehouse = warehouse
        
        # Coordinates
        self.bottom_left_corner = np.array((  int(np.ceil(level/2-0.5))*alley_width + level*shelve_width,0 ))
        shelve_height = self.n_spots*0.5
        
        # Creating spots (they need the coordinates)
        for spot in range(1,self.n_spots+1):
            # see notes.md for details
            acccess_time = (0.5*spot + self.bottom_left_corner)  * unit_drive_time # (height + width) * unit drive time
            spot_corner = self.bottom_left_corner+np.array((0,0.5*(spot-1)))
            self.spots.append(Spot(acccess_time, spot, shelve = self, size=spot_size, coordinates=spot_corner))

        
        # Rendering box
        self.shelve_box = Rectangle( self.bottom_left_corner, shelve_width, shelve_height ,edgecolor='k',facecolor='none')


    def weight(self):
        ''' Compute the cumulative weight of the shelve
        '''
        return sum([spot.package.weight if spot.is_full else 0 for spot in self.spots])

    def moving_energy_cost(self):
        '''
        Compute cost in dollar for moving the aisle given its weight and physical properties - used to compute reward
        :return: Float -
        '''
        return energy_cost_fn(self.weight())

    def __str__(self):
        self.describe()


    def describe(self, prefix = '', cascade = True):
        ''' Print a verbose description of the shelve
        '''
        print(prefix,'Shelve of level {} with {} spots'.format(self.level,self.n_spots))
        print(prefix,'My spots are : ')
        if cascade:
            for spot in self.spots:
                spot.describe(prefix+'   ')


    def plot(self, ax = None, verbose = 0):
        ''' Plot the shelve on an ax using matplotlib rectangles.

        If an ax is given, the rectangle will be added to it and the plot will not be rendered.
        '''
        render = False

        if ax is None:
            render = True
            fig, ax = plt.subplots(1)
            ax.set_title('Shelve')
            
        ax.add_patch(self.shelve_box)

        if verbose > 0:
            for spot in self.spots:
                spot.plot(ax,self.bottom_left_corner,verbose=verbose-1)

        if render:
            plt.show()
            
    def get_patch(self):
        return self.shelve_box
    
    def get_patches(self,spots=True, unrendered_only=False):
        
        if spots :
            spot_patches = []
            for spot in self.spots:
                spot_patch = spot.get_patch(unrendered_only=unrendered_only)
                if spot_patch:
                    if isinstance(spot_patch,list):
                        spot_patches += spot_patch
                    else:
                        spot_patches.append(spot_patch)
            return self.get_patch(),spot_patches
        else:
            return self.get_patch(),None
  