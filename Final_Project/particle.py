import numpy as np
import time
import matplotlib.pyplot as plt
from grid import *
from ode import *
class Particle():
    """Desrcibes the porperty of one particle in the n-bopdy simulation"""

    def __init__(self, mass, x, y,px=0.0,py=0.0):
       self.mass = mass
       self.position = np.array([x,y])
       self.momentum = np.array([px,py])

                


class ParticleList():
    """List to store a collection of particle and being able to call their attricutes directly"""
    def __init__(self,nparticles,initial_condition):
        self.nparticles = nparticles
        self.particles = np.array([Particle(mass,x,y) for x,y,mass in initial_condition])
        self.mass = np.array([self.particles[i].mass for i in range(len(self.particles))])
        self.position = np.array([self.particles[i].position for i in range(len(self.particles))])
        self.momentum = np.array([self.particles[i].momentum for i in range(len(self.particles))])

    def __len__(self):
        return self.nparticles

    def evolve(self,acc,acc_new, dt, size, boundary_periodic=True):
            # print(self.position.shape)
            # print(self.momentum.shape)
            # print(acc.shape)
            # print(acc_new.shape)
            self.position, self.momentum = leap_frog(self.position, self.momentum, acc, acc_new, dt)
            if boundary_periodic == True:
                self.position=self.position%(size-1)
            




class NBodySystem():
    """Descrribes the whole system of the nbody particles"""

    def __init__(self,nparticles,size, mass=1, G=1, boundary_periodic = True, softner=0.1, dt = 1, position = None, momentum = None):
        self.parameters = {}
        self.parameters['softener'] = softner
        self.parameters['G'] = G
        self.parameters['boundary_periodic'] = boundary_periodic
        self.parameters['nparticles'] = nparticles
        self.size = size
        self.mass = mass
        self.dt = dt
        if boundary_periodic==True:
            self.grid_size = size
        else:
            self.grid_size = 4*size
        self.ptclgrid = ParticleGrid(nparticles,self.grid_size,self.size, mass=mass)
        self.grid = self.ptclgrid.grid
        self.grid_pos = self.ptclgrid.grid_pos
        mass = np.ones(nparticles)*1
        x0,y0 = self.ptclgrid.position.transpose()
        initial_condition = np.array([x0,y0, mass]).transpose()
        self.particles = ParticleList(nparticles, initial_condition)
        self.compute_green_function(self.grid_size)
        self.acc = np.zeros((len(self),2))


    def __len__(self):
        return len(self.particles.position[:,0])

    def compute_green_function(self,n):
        size = np.arange(n)
        xx,yy = np.meshgrid(size,size)
        vectors = np.array([xx.ravel(),yy.ravel()])
        norm = norm_on_grid(vectors)
        green = green_function(norm,self.grid,self.parameters['softener'],self.parameters['G'])
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
        phi = np.real(np.fft.ifft2(np.fft.fft2(self.grid)*np.fft.fft2(self.green)))
        self.phi = phi[:self.size,:self.size]
        return phi

    def grad_phi_mesh(self, phi):
        if self.parameters['boundary_periodic'] == True:
            fx = -0.5 * (np.roll(phi, 1, axis = 1) - np.roll(phi, -1, axis=1)) 
            fy = -0.5 * (np.roll(phi, 1, axis = 0) - np.roll(phi, -1, axis=0)) 
        else:
            fx = -0.5 * (phi[2:,1:-1] - phi[0:-2,1:-1])
            fx = np.insert(fx,0,0, axis=0)
            fx = np.insert(fx,-1,0,axis=0)
            fx = np.insert(fx,0,0, axis=1)
            fx = np.insert(fx,-1,0,axis=1)
            fy = -0.5 * (phi[1:-1,2:] - phi[1:-1,:-2])
            fy = np.insert(fy,0,0, axis=0)
            fy = np.insert(fy,-1,0,axis=0)
            fy = np.insert(fy,0,0,axis=1)
            fy = np.insert(fy,-1,0,axis=1)
        return fx,fy
    
    def compute_forces_mesh(self, phi):
        f = self.ptclgrid.grid*self.grad_phi_mesh(phi)
        return f
                     

    def energy(self):
        energy = -0.5*np.sum(self.phi)+0.5*self.mass*np.sum(self.particles.momentum)**2
        return energy

    def evolve_system(self):
        phi = self.compute_field()
        force_m = self.compute_forces_mesh(phi)
        self.acc_new = np.zeros([len(self),2])
        for i in range(len(self)):
            x,y = self.ptclgrid.ixy[i]
            x = int(x)
            y = int(y)
            self.acc_new[i][0] += 1/self.mass*force_m[0][x,y]
            self.acc_new[i][1] += 1/self.mass*force_m[1][x,y]
        self.particles.evolve(self.acc,self.acc_new,self.dt,self.size, boundary_periodic=self.parameters['boundary_periodic'])
        if self.parameters['boundary_periodic']!=True:
            index = np.argwhere(~(self.particles.position<=self.grid_size-1))
            index2 = np.argwhere(~(self.particles.position>=0))
            index = {a for a in np.append(index,index2)}
            index = list(index)
            self.particles.momentum = np.delete(self.particles.momentum,index,axis=0)
            self.acc = np.delete(self.particles.position,index,axis=0)
            self.acc_new = np.delete(self.particles.position,index,axis=0)
            self.particles.position = np.delete(self.particles.position,index,axis=0)
        self.acc = self.acc_new.copy()
        self.ptclgrid.update_position(self.particles.position)
        self.grid = self.ptclgrid.grid
        self.grid_pos = self.ptclgrid.grid_pos
        print(self.energy())
        #return forces



if __name__=='__main__':
    plt.ion()
    n=int(1e5)
    size=400
    system = NBodySystem(n,size, mass=1, boundary_periodic = True)
    #f = system.evolve_system()


    for i in range(0,10000):
        #for ii in range(5):
        system.evolve_system()
        print('One step')
        plt.clf()
        plt.pcolormesh(system.grid_pos[:size,:size])
        plt.colorbar()
        #print(system.particles.position[:,0],system.particles.position[:,1])
        plt.pause(1e-3)

