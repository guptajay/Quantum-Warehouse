import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from warehouse.envs.package import Package
from warehouse.envs.constants import frequency_cmap, spot_size
from warehouse.envs.utils import id_counter, spot_potential_package_ids

class Spot():
    ''' Describe a single spot (only holds one object!)
    If no shelve is given, we consider that it is the package to drop!
    '''
    
    def __init__(self,access_time, position, shelve, size=spot_size, coordinates=None):

        self.position = position
        self.package = []
        self.size = size
        self.is_full = self._is_full()
        self.shelve = shelve
        self.id = id_counter()

        self.access_time = access_time
        self.access_cost = lambda : self.shelve.warehouse.compute_access_cost(self)

        # Rendering states
        
        if coordinates is not None:
            self.spot_corner = coordinates
        elif self.shelve:
            shelve_corner = self.shelve.bottom_left_corner
            self.spot_corner = shelve_corner + np.array(  ( 0.5*(self.shelve.n_spots - self.position) if self.shelve.orientation == 1 else 0,   0.5*(self.shelve.n_spots - self.position) if self.shelve.orientation == 0 else 0)  )
        else:
            raise Exception('Please give a parent shelve or specify coordinates')
            
        self.rendered = False
            
    # Actions

    def put(self, package):
        if package.id not in spot_potential_package_ids(self.size, self.package) and not self.size == -1:
            raise Exception('Could not put object on a filled shelve')
        self.package.append(package)
        self.rendered = False
        
    def take(self, package_id):
        for i, package in enumerate(self.package):
            if package.id == package_id:
                self.package.pop(i)
                self.rendered = False
                return package
        raise Exception('Package id not available in this spot')

    def _is_full(self):
        return len(self.package) == self.size

    def empty(self):
        self.package = []

    def is_empty(self):
        return len(self.package)==0


    # Utils

    ## Render
    def get_patch(self, unrendered_only = False):
        if unrendered_only and self.rendered:
            return None
        
        self.rendered = True

        patches = [Rectangle( (self.spot_corner[0],self.spot_corner[1]), 1/2 , 1/2 ,edgecolor='k', facecolor= 'none')]


        if len(self.package)>0:

            n_rows_cols = int(np.ceil(np.sqrt(len(self.package))))
            
            def get_pos(index):
                col = index%n_rows_cols
                row = index//n_rows_cols
                return (row,col)

            increment = 0.5/n_rows_cols

            for package,index in zip(self.package,range(len(self.package))):
                i,j = get_pos(index)
                patches.append(Rectangle( self.spot_corner+(increment*i,increment*j), increment , increment ,edgecolor='k', facecolor= frequency_cmap(package.frequency)))            

        return patches

    def describe(self,prefix = ''):
        ''' Print a verbose description of the spot
        '''
        print(prefix,'Spot with position {}, access time {} and packages {}'.format(self.position,self.access_time,str(self.package)[1:-1] if self.is_full else 'None'))

    def __str__(self):
        self.describe()
