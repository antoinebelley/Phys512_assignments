import numpy as np
from numba import *


class ParticleGrid(object):
    """Class to construct the density grid with the particle"""
    def __init__(self,nparticle,grid_size,size,mass=1.0, dim=2):
        """Initialize the class.
        -Arguments: -nparticles (int): The number of particle to run in the simulation
                    -size       (int): Size of the grid
                    -dim        (int): Dimension of the grid. Default = 2
        - Fields:   -position (array): The position of the particles. It is initialized as random
                    -grid     (array): The grid on which we place the particle
                    -ixy      (array): Position of the particle as integers (for the density)
        -Returns:   -void

        """
        self.position = np.random.rand(nparticle,dim)*(size-1)
        self.grid = np.zeros((grid_size,grid_size), dtype = np.float64)
        self.grid_pos = np.zeros((grid_size,grid_size), dtype = np.float64)
        self.ixy_pos = np.asarray(np.rint(self.position),dtype=np.int64)
        self.ixy = np.asarray(np.floor(self.position),dtype=np.int64)
        self.ixy = np.append(self.ixy,np.asarray(np.ceil(self.position),dtype=np.int64),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.ceil(self.position[:,0]),np.floor(self.position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.floor(self.position[:,0]),np.ceil(self.position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.mass = mass
        self.hist_2D()

    def hist_2D(self):
        """Creates a 2D histogramm with the density of the particles by adding one to every index where there is a particle on the grid"""
        self.grid = 0*self.grid
        self.grid_pos = 0*self.grid_pos
        n = self.ixy.shape[0]
        for i in range(n):
            self.grid[self.ixy[i,0],self.ixy[i,1]]+=self.mass/4
        for i in range(self.ixy_pos.shape[0]):
            self.grid_pos[self.ixy_pos[i,0],self.ixy_pos[i,1]]+=self.mass

    def update_position(self,position):
        """Update the position of the particles on the grid"""
        #self.ixy = np.asarray(np.abs(np.rint(position)),dtype=np.int64)
        self.ixy = np.asarray(np.floor(position),dtype=np.int64)
        self.ixy = np.append(self.ixy,np.asarray(np.ceil(position),dtype=np.int64),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.ceil(position[:,0]),np.floor(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.floor(position[:,0]),np.ceil(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy_pos = np.asarray(np.rint(position),dtype=np.int64)
        self.hist_2D()
       

@njit
def green_function(norms,grid,soft,G):
    soft = soft**2
    norms[norms<soft] = soft
    norms=norms
    green = np.ones(grid.size)
    green = green/norms
    green = green.reshape(grid.shape)
    green = green.transpose()
    return green

@njit
def norm_on_grid(raveled):
    norm = np.sqrt(raveled[0,]**2+raveled[1,]**2)
    return norm


# grid = ParticleGrid(100,1000).grid
# size = np.arange(1000)
# xx,yy = np.meshgrid(size,size)
# vectors = np.array([xx.ravel(),yy.ravel()])
# norm = norm_on_grid(vectors)
# green = green_function(norm,grid,1000,0.1,1)
# import matplotlib.pyplot as plt
# plt.imshow(grid)
# plt.show()
# green_function(100,grid,0.01,1,xx,yy)