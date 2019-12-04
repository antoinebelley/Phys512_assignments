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
        self.update_grid()

    def update_grid(self):
        """Creates a 2D histogramm with the density of the particles by adding one to every index where there is a particle on the grid"""
        
        self.grid, self.grid_posd = hist_2D(self.grid, self.grid_pos,self.ixy,self.ixy_pos,self.mass)
        # self.grid = 0*self.grid
        # self.grid_pos = 0*self.grid_pos
        # n = self.ixy.shape[0]
        # for i in range(n):
        #     self.grid[self.ixy[i,0],self.ixy[i,1]]+=self.mass/4
        # for i in range(self.ixy_pos.shape[0]):
        #     self.grid_pos[self.ixy_pos[i,0],self.ixy_pos[i,1]]+=self.mass

    def update_position(self,position):
        """Update the position of the particles on the grid"""
        #self.ixy = np.asarray(np.abs(np.rint(position)),dtype=np.int64)
        self.position = position
        self.ixy = np.asarray(np.floor(position),dtype=np.int64)
        self.ixy = np.append(self.ixy,np.asarray(np.ceil(position),dtype=np.int64),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.ceil(position[:,0]),np.floor(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.floor(position[:,0]),np.ceil(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy_pos = np.asarray(np.rint(position),dtype=np.int64)
        self.update_grid()
       

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

@njit
def hist_2D(grid, grid_pos,ixy,ixy_pos,mass):
    """Creates a 2D histogramm with the density of the particles by adding one to every index where there is a particle on the grid"""
    grid = 0*grid
    grid_pos = 0*grid_pos
    n = ixy.shape[0]
    for i in range(n):
        grid[ixy[i,0],ixy[i,1]]+=mass/4
    for i in range(ixy_pos.shape[0]):
        grid_pos[ixy_pos[i,0],ixy_pos[i,1]]+=mass
    return grid, grid_pos