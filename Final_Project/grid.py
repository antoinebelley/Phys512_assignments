import numpy as np
from numba import *


class ParticleGrid(object):
    """Class to construct the density grid with the particle"""
    def __init__(self,nparticle,grid_size,size,mass, dim=2):
        """Initialize the class.
        -Arguments: -nparticles (int): The number of particle to run in the simulation
                    -size       (int): Size of the grid
                    -dim        (int): Dimension of the grid. Default = 2
        - Fields:   -position (array): The position of the particles. It is initialized as random
                    -grid     (array): The grid on which we place the particle
                    -ixy      (array): Position of the particle as integers (for the density)"""
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
        """Creates a 2D histogramm with the density of the particles by adding using a CIC like model. For
        perfomrance purposes, it calls the compiled fucnction hist_2D which is compiled using numba"""
        self.grid, self.grid_pos = hist_2D(self.grid, self.grid_pos,self.ixy,self.ixy_pos,self.mass)

    def update_position(self,position, mass):
        """Update the position of the particles on the grid
        -Arguments: -position (array): The new positions of the particle on the grid"""
        self.mass
        self.position = position
        self.ixy = np.asarray(np.floor(position),dtype=np.int64)
        self.ixy = np.append(self.ixy,np.asarray(np.ceil(position),dtype=np.int64),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.ceil(position[:,0]),np.floor(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy = np.append(self.ixy, np.asarray([np.floor(position[:,0]),np.ceil(position[:,1])],dtype=np.int64).transpose(),axis=0)
        self.ixy_pos = np.asarray(np.rint(position),dtype=np.int64)
        self.update_grid()

    def early_universe_grid(self):
        """Creates a 2D histogramm with the density of the particles by adding using a CIC like model and the distribution from the early universe."""
        norms = norm_on_grid(self.ixy_pos.transpose())
        norms[norms>1000]=1
        mass = (norms**(-3))
        self.mass = mass
        self.grid, self.grid_pos = hist_2D(self.grid, self.grid_pos,self.ixy,self.ixy_pos,self.mass)
        


       

@njit
def green_function(norms,grid,soft,G):
    """Computes the green function on a grid.
    -Arguments: -norms (array): The distance of each point from the origin of the grid
                -grid  (array): The grid onto which take the green function
                -soft  (float): Softner to apply to avoid infinity
                -G     (float): The value of Newton's we wish to use
    -Retuns: -green (array): The green function on the grid"""
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
    """Computes the norm of a list of vector
    -Arguments: - raveled (array): Array containing the x and y coord of each vector
    -Returns: - norm (array): Array containing the norm of each vector """
    norm = np.sqrt(raveled[0,]**2+raveled[1,]**2)
    return norm


@njit
def hist_2D(grid, grid_pos,ixy,ixy_pos,mass):
    """Creates a 2D histogramm with the density of the particles by adding one to every index where there is a particle on the grid
    and one using the CIC like model.
    -Arguments: -grid     (array): Grid where to place the densities of the masses
                -grid_pos (array): Grid with the position of the particles
                -ixy      (array): Corner of the cells where the particles are
                -ixy_pos  (array): Position of the nearest grid cell to particle positions
                -mass     (array): Mass of the particles
    -Returns: - grid    (array): Updated grid with the density of the particles
              - grid_pos(array): Updated grid with the positions of the particles """
    grid = 0*grid
    grid_pos = 0*grid_pos
    n = ixy.shape[0]
    for i in range(ixy_pos.shape[0]):
        grid_pos[ixy_pos[i,0],ixy_pos[i,1]]+=mass[i]
        for j in range(4):
            grid[ixy[4*i+j,0],ixy[4*i+j,1]]+=mass[i]/4
    return grid, grid_pos