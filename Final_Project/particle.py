import numpy as np
import time
import matplotlib.pyplot as plt
from grid import *
import time
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

    def evolve(self,force, dt):
            self.position += (self.momentum*dt)
            self.momentum += (force*dt)
            




class NBodySystem():
    """Descrribes the whole system of the nbody particles"""

    def __init__(self,nparticles,size, mass=1, G=1, boundary_periodic = True, softner=0.5):
        self.parameters = {}
        #self.parameters['tfinal'] = tfinal
        self.parameters['softener'] = softner
        self.parameters['G'] = G
        self.parameters['boundary_periodic'] = boundary_periodic
        self.parameters['nparticles'] = nparticles
        self.ptclgrid = ParticleGrid(nparticles,size, mass=mass)
        self.grid = self.ptclgrid.grid
        self.size = size
        mass = np.ones(nparticles)*mass
        x0,y0 = self.ptclgrid.position.transpose()
        initial_condition = np.array([x0,y0, mass]).transpose()
        self.particles = ParticleList(nparticles, initial_condition)
        self.compute_green_function(size)
    
    def __len__(self):
        return self.parameters['nparticles']

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
        phi = np.abs(np.fft.ifft2(np.fft.fft2(self.grid)*np.fft.fft2(self.green)))
        self.phi = phi
        return phi

    def grad_phi_mesh(self, phi):
        f = np.gradient(phi)
        return f
    
    def compute_forces_mesh(self, phi):
        f = self.grid*self.grad_phi_mesh(phi)
        return f
                     

    def evolve_system(self):
        phi = self.compute_field()
        force_m = self.compute_forces_mesh(phi)
        forces = np.zeros([len(self),2])
        for i in range(len(self)):
            x,y = self.ptclgrid.ixy[i]
            x = int(x)%self.size
            y = int(y)%self.size
            forces[i][0] = -force_m[0][x,y]
            forces[i][1] = -force_m[1][x,y]
        self.particles.evolve(forces,0.1)
        self.ptclgrid.update_position(self.particles.position-1)

        self.grid = self.ptclgrid.grid
        print(forces)
        return forces



if __name__=='__main__':
    plt.ion()
    n=int(1e5)
    size=500
    system = NBodySystem(n,size, boundary_periodic = True)
    #system.compute_field()
    # size = np.arange(size)
    # xx,yy=np.meshgrid(size,size)
    # phi = system.compute_field()
    # f = system.evolve_system(phi)
    # f=f.transpose()
    # plt.pcolormesh(xx,yy,phi)
    # plt.ylim(0,yy.ravel()[-1])
    # pos= system.particles.position.transpose()
    #plt.quiver(pos[0], pos[1], -f[0],-f[1])
    # plt.show()
    # system.evolve_system()
    # plt.plot(system.particles.position[:,0],system.particles.position[:,1],'*')
    # plt.show()

    for i in range(0,10000):
        #for ii in range(5):
        system.evolve_system()
        print('One step')
        plt.clf()
        plt.imshow(system.grid)
        #print(system.particles.position[:,0],system.particles.position[:,1])
        plt.pause(1e-3)

