import numpy as np
import time
import matplotlib.pyplot as plt
from grid import *
from ode import *


class Particle():
    """Desrcibes the porperty of one particle in the n-bopdy simulation 
       -Fieds: - mass     (float): The mass of the particle
               - position (array): The position of the particle in x,y
               - momentum (array): The momentum of the particle in x,y 

       -Methods: __init__() : Initiate the class and populate the attributes """
    def __init__(self, mass, x, y,px=0.0,py=0.0):
       """Initialize the Particle class and populate the attributes
       -Arguments: -mass (float) : the mass of the particle
                   -x    (float) : The x-coordinate of the particle
                   -y    (float) : The y-coordinate of the particle
                   -px   (float) : The momentum of the partilce in x-coordinate. Default is 0
                   -py   (float) : The momentum of the particle in y-coordinate. Default is 0."""
       self.mass = mass
       self.position = np.array([x,y])
       self.momentum = np.array([px,py])

                
class ParticleList():
    """List to store a collection of particle and being able to call their attributes simultaneously
       -Fields: -nparticles   (int): The number of partciles in the list
                -particles  (array): Individual instances of Particle class in the list
                -mass       (array): Mass attribute of the Particle instances in particles field
                -positon    (array): Position attribute of the Particle instances in particles field
                -momentum   (array): Momentum attribute of the Particle instances in particles field

        -Methods: __init__() : Initiate the class and populate the attributes
                  __len__()  : Returns the length of the list
                  evolve()   : Computes the new particle and position of the particles for a step of dt"""
    def __init__(self,nparticles,initial_condition):
        """Initialize the ParticleList class and populates the attributes
           -Arguments: nparticles         (int): Number of the particles to be sotred in the list
                       initial_conditon (array): Array containing the initial position and mass of the particles. The length needs
                                                 to be equal as nparticles."""
        self.nparticles = nparticles
        self.particles = np.array([Particle(mass,x,y) for x,y,mass in initial_condition])
        self.mass = np.array([self.particles[i].mass for i in range(len(self.particles))])
        self.position = np.array([self.particles[i].position for i in range(len(self.particles))])
        self.momentum = np.array([self.particles[i].momentum for i in range(len(self.particles))])

    def __len__(self):
        """Computes the lenght of the list of particles
           -Return: self.nparticles (int): The lenght of the list"""
        return self.nparticles

    def evolve(self,acc,acc_new, dt, size, boundary_periodic=True):
        """Evolve the positions and momentum of the particles in the list using the leap frog method. 
           -Arguments: -acc              (array): Array of the current acceleration of the particles
                       -acc_new          (array): Array of the new acceleration of the particles (after being evolved in NBodySystem)
                       -dt               (float): The size of the time step to be used in the leap frog method to solve the ode
                       -size               (int): The size of the grid on which the partilces are. Used to create a toroïdal space for the periodic boundary condition
                       -boundary_periodic (bool): Condition to set boundary condition. If True (i.e. periodic boundary condition) the particles are brought back at the opposite
                                                  side of the grid, i.e. we have a toroïdal space."""
        self.position, self.momentum = leap_frog(self.position, self.momentum, acc, acc_new, dt)
        if boundary_periodic == True:
            self.position = self.position%(size-1)


class NBodySystem():
    """Describes the whole system of the nbody particles. To do so, it convolves the density with the green function on the grid to find
    the potental. It then takes the gradient of the potential to find the force on each grid cell and is then interpolated to the particle
    to find the force on them. Density assignment is done using a method similar to the cloud in cell model. However the method is simplify by simply assessing 
    a quarter of the mass to the four corner in which the particle lies.
    -Fields: - softener               (float): The softner to apply on the green function.
             - G                      (float): Value of the gravitational constant
             - boundary_periodic       (bool): Condition to set boundary condition to periodic.
             - nparticles               (int): The number of particles in the sytsem
             - size                     (int): Size of the grid where the particles lie
             - mass                   (float): The mass of the particles
             - grid_size                (int): The size of the grid on which the potential is computed. For non persiodic boundary condition, this is twice size
             - ptclgrid        (ParticleGrid): Instance of the class ParticleGrid used to construct the grid
             - grid                   (array): The grid containing the particle density
             - grid_pos               (array): Grid containing the particles position
             - particles       (ParticleList): Instance of the class ParticleList that contains the position and momentum of the particles in the system
             - acc                    (array): Array containing the current acceleration of the particles
             - green                  (array): The green function on the grid.
             - phi                    (array): Potential on the grid. ONLY INITIALIZED BY RUNNING METHOD compute_field
             - acc_new                (array): New acceleration of the particle after updating the particle's positions and momenta. 
                                               ONLY INITIALIZED BY RUNNING METHOD evolve

    -Methods: __init__()               : Initiate the class and populate the attributes
              __len__()                : Returns the length of the system
              compute_green_function() : Computes the green function on the grid. This is called in _init_ and should not be called afterwards.
              compute_field()          : Computes the field on the grid by taking the convolution of the density and the green function.
              grad_phi_mesh()          : Takes the gradient of the field in each case of the grid
              computes_forces_mesh()   : Interpolate the force on the grid with particles position
              energy()                 : Compute the total energy of the system
              evolve_system()          : Compute the new accelerations of the system by finding the force and update the particles positions."""

    def __init__(self,nparticles,size, mass=1, G=1, boundary_periodic = True,early_universe=False, softner=1, position = [], momentum = []):
        """Initialize the NBodySystem class and populates the attributes
           -Arguments: -nparticles         (int): Number of the particles to be sotred in the list
                       -size               (int): Size of the grid for the system
                       -mass             (float): Masses of the particles in the system. Default = 1
                       -G                (float): Value of Newton's constant. Default =1
                       -early_universe    (bool): Set the mass distribution to k^-3 if true. Default = True
                       -boudary_periodic  (bool): Condition to set the boundary condition on the system. Default = True
                       -softener         (float): The softener for the green function to avoid infinite values. Default = 1
                       -position         (array): Initial position of the particles. If not is given, they will be
                                                  initialized at random on the grid.
                       -momentum         (array): Initial momentum of the partciles. If none is given, they will be
                                                  initialized to 0."""
        self.softner = softner
        self.G = G
        self.boundary_periodic = boundary_periodic
        self.nparticles = nparticles
        self.size = size
        self.mass = np.ones(nparticles)*mass
        #If the boundary condition are not periodic, the grid_size is double but particle kept in the first quadrant so 
        #that the particles cannot feel the effect of the particles closed to the opposite boundary when we take the convolution
        if boundary_periodic==True:
            self.grid_size = size
        else:
            self.grid_size = 2*size
        #Initialize the partticle grid
        # if early_universe == True:
        #     self.ptclgrid.early_universe_grid(softner)
        #     self.mass = self.ptclgrid.mass
        self.ptclgrid = ParticleGrid(nparticles,self.grid_size,self.size, mass=self.mass, soft=softner, early_universe=early_universe)
        #If initial position are givem, place the particle to the right place on the grid
        if len(position) != 0:
            self.ptclgrid.update_position(position, mass)

        self.grid = self.ptclgrid.grid
        self.grid_pos = self.ptclgrid.grid_pos
        x0,y0 = self.ptclgrid.position.transpose()
        initial_condition = np.array([x0,y0, self.mass]).transpose()
        #Initialize the Particle list containing the position and momentum of the particles
        self.particles = ParticleList(nparticles, initial_condition)
        #If initial mometa are given, intialize it 
        if len(momentum) != 0:
            self.particles.momentum = momentum
        #Computes the green function on the grid
        self.compute_green_function(self.grid_size)
        #Initialize the array with the acceleration of the particles
        self.acc = np.zeros((len(self),2))

    def __len__(self):
        """Computes the lenght of the list of particles
           -Return: len(self.particles.position[:,0]) (int): The lenght of the list of particles"""
        return len(self.particles.position[:,0])

    def compute_green_function(self,n):
        """Comoutes the green function on the grid. Uses the function norm and green_function from grid.py to compute it. This function are compiled using 
        the JIT compiler Numba to improve performance. It then make the green function periodic to be able to work for the convolution, i.e it places the green 
        function in the four corners of the grid.
        -Arguments: - n     (int): The size of the grid on which we want to compute the potential"""
        size = np.arange(n)
        xx,yy = np.meshgrid(size,size)
        vectors = np.array([xx.ravel(),yy.ravel()])
        norm = norm_on_grid(vectors)
        green = green_function(norm,self.grid,self.softner,self.G)
        try:
            green[n//2:, :n//2] = np.flip(green[:n//2, :n//2],axis=0)
            green[:n//2, n//2:] = np.flip(green[:n//2, :n//2],axis=1)
            green[n//2:, n//2:] = np.flip(green[:n//2, :n//2])
        except:
            green[n//2:, :n//2+1] = np.flip(green[:n//2+1, :n//2+1],axis=0)
            green[:n//2+1, n//2:] = np.flip(green[:n//2+1, :n//2+1],axis=1)
            green[n//2:, n//2:] = np.flip(green[:n//2+1, :n//2+1])
        self.green = green

    def compute_field(self):
        green_fft = np.fft.fft2(self.green)
        """Compute the field on the grid by taking the convolution of the denisty on the grid and the green fucntion."""
        phi = np.real(np.fft.ifft2(np.fft.fft2(self.grid)*green_fft))
        self.phi = phi[:self.size,:self.size] #This is the green function only where particle lives. Only useful for non-periodic bc
        if self.boundary_periodic != True:
          self.phi[0:,0]=0
          self.phi[0:,-1]=0
          self.phi[-1:,0]=0
          self.phi[-1,-1]=0
        return phi

    def grad_phi_mesh(self):
        """Computes the gradient of the field on the grid. We use the central difference to take the gradient.
        -Arguments: - phi (array): The potential of which we want to take the gradient of
        -Returns: - fx (array): The forces in the x-coordinates on each grid cells
                  - fy (array): The forces in y-coordinates on each grid cells."""
        fy = -0.5 * (np.roll(self.phi, 1, axis = 1) - np.roll(self.phi, -1, axis=1)) 
        fx = -0.5 * (np.roll(self.phi, 1, axis = 0) - np.roll(self.phi, -1, axis=0))
        return fx,fy
    
    def compute_forces_mesh(self):
        """Interpolates the force to the masses
        -Arguments: - phi (array): The potential of which we want to take the gradient of
        -Returns: - f (array)"""
        f = self.ptclgrid.grid[:self.size,:self.size]*self.grad_phi_mesh()
        return f
                     
    def energy(self):
        """Computes the energy of the system using E = V+0.5*mv**2
        -Returns: - energy (float): The energy of the system."""
        energy = -0.5*np.sum(self.phi)+0.5*np.sum(self.mass*np.sqrt(self.particles.momentum[:,0]**2+self.particles.momentum[:,1]**2)**2)
        return energy

    def evolve_system(self,dt, energy_file = None):
        """Evolves the acceleration, position and momentum of the particles.
        -Arguments: -dt         (float): Time step to take in the evolution
                    -energy_file (file): File where to save the energy at each step if given. Default = None.
        -Returns: -grid_pos  (array): Array with the positions of the particles on the grid"""
        phi = self.compute_field()
        force_m = self.compute_forces_mesh()
        self.acc_new = np.zeros([len(self),2])
        #Computes the force felt by each particles and deduce the acceleration
        for i in range(len(self)):
            x,y = self.ptclgrid.ixy[i]
            x = int(x)
            y = int(y)
            self.acc_new[i][0] += (1/self.mass[i]*force_m[0][x,y])
            self.acc_new[i][1] += (1/self.mass[i]*force_m[1][x,y])
        #Evolve the position and momenta of the particle in the list
        self.particles.evolve(self.acc,self.acc_new,dt,self.size, boundary_periodic=self.boundary_periodic)
        #For non-periodic condition, deletes the particles that leave the grid from the list
        if self.boundary_periodic!=True:        
            index = np.argwhere((self.particles.position>self.size-1))
            index2 = np.argwhere((self.particles.position<0))
            index = {a for a in np.append(index,index2)}
            index = list(index)
            self.particles.momentum = np.delete(self.particles.momentum,index,axis=0)
            self.acc = np.delete(self.acc,index,axis=0)
            self.acc_new = np.delete(self.acc_new,index,axis=0)
            self.mass = np.delete(self.mass,index,axis=0)
            self.particles.position = np.delete(self.particles.position,index,axis=0)
        self.acc = self.acc_new.copy()
        #Update the position of the particles on the grid
        self.ptclgrid.update_position(self.particles.position,self.mass)
        self.grid = self.ptclgrid.grid
        self.grid_pos = self.ptclgrid.grid_pos
        #Write the energy in a file if on is given
        if energy_file != None:
            energy_file.write(f'{self.energy()}\n')
            energy_file.flush()
        return self.grid_pos