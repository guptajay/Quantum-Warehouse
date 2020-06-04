import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from warehouse.envs.package import Package
from warehouse.envs.constants import frequency_cmap
from warehouse.envs.utils import id_counter

class Spot():
    ''' Describe a single spot (only holds one object!)
    If no shelve is given, we consider that it is the package to drop!
    '''
    
    def __init__(self,access_time, position, shelve, coordinates = None):
        

        self.position = position
        self.is_full = False
        self.package = None
        self.shelve = shelve
        self.id = id_counter()

        self.access_time = access_time
        self.access_cost = lambda : self.shelve.warehouse.compute_access_cost(self)

        # Rendering states
        
        if coordinates:
            self.spot_corner = coordinates
        elif self.shelve:
            shelve_corner = self.shelve.bottom_left_corner
            self.spot_corner = shelve_corner + np.array(  ( 0.5*(self.shelve.n_spots - self.position) if self.shelve.orientation == 1 else 0,   0.5*(self.shelve.n_spots - self.position) if self.shelve.orientation == 0 else 0)  )
        else:
            raise Exception('Please give a parent shelve or specify coordinates')
            
        self.rendered = False
            
    # Actions

    def put(self,package):
        if self.is_full:
            raise Exception('Could not put object on a filled shelve')
        self.package = package
        self.is_full = True
        self.rendered = False
        
    def take(self):
        package = self.package
        self.is_full = False
        self.package = None
        self.rendered = False
        return package

    # Utils

    ## Render
    def get_patch(self, unrendered_only = False):
        if unrendered_only and self.rendered:
            return None
        
        self.rendered = True
        return Rectangle( self.spot_corner, 1/2 , 1/2 ,edgecolor='k', facecolor= frequency_cmap(self.package.frequency) if self.is_full else 'none')

    def describe(self,prefix = ''):
        ''' Print a verbose description of the spot
        '''
        print(prefix,'Spot with position {}, access time {} and package {}'.format(self.position,self.access_time,str(self.package) if self.is_full else 'None'))

    def __str__(self):
        self.describe()

