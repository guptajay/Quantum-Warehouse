import logging
import threading
import time
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

from warehouse.envs.constants import plt

class Renderer():
    ''' A renderer for the gym environment
    
    # Args
        - fig_shape : ( (min_x,max_x), (min_y,max_y) ) for the figure bounds
        - shelves : patches for the shelves
        - spots : patches for the initial spots
    '''
    
    def __init__(self, fig_shape, shelves, spots, stats = None):
    
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.ax.set_title('Warehouse')
        
        # Autoscale on unknown axis and known lims on the other
        #self.ax.set_autoscaley_on(True)
        self.shape = fig_shape
        self.ax.set_xlim(*self.shape[0])
        self.ax.set_ylim(*self.shape[1])
        self.ax.set_aspect('equal')
        
        # Patches and texts
        self.shelves = shelves
        self.spots = { spot.xy : spot for spot in spots }
        
        # Draw the shelves (only once)
        for shelve in self.shelves:
            self.ax.add_patch(shelve)
            
        # Draw the initial spots
        for spot in self.spots.values():
            self.ax.add_patch(spot)
                
        # Stats
        self.stats = stats
        self.stats_patch = None
        self.draw_stats()
        
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, spots):
        ''' Update the spots
        
        #Args
            - spots : list of Rectangle patches.
        '''
    
        # Remove old spots and add new ones
        for spot in spots:
            if spot.xy in self.spots:
                self.spots[spot.xy].remove()
            else:
                print('WARNING : renderer received a new box a {}'.format(spot.xy))
                
            self.spots[spot.xy] = spot
            self.ax.add_patch(spot)
        
        self.draw_stats()
        
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
    def draw_stats(self):
        if self.stats:
            
            if self.stats_patch:
                self.stats_patch.remove()
            
            self.stats_patch = None
            
            x_margin,y_margin = 0.2, -1.5
            stats = '\n'.join([ '{}:{}'.format(stat,self.stats[stat]) for stat in self.stats ])
            self.stats_patch = self.ax.text(self.shape[0][0]+x_margin,self.shape[1][1]+y_margin,stats)
    
    def close(self):
        input('Script is ended. Press any key to exit the viewer')