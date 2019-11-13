import numpy as np
import time
import matplotlib.pyplot as plt
from mesh import MeshGrid

class Particle():
    """Desrcibes the porperty of one particle in the n-bopdy simulation"""

    def __init__(self, mass, x, y,z,px=0.0,py=0.0,pz=0.0, limit=10):
       self.mass = mass
       self.position = np.array([x,y,z])%limit
       self.momentum = np.array([px,py,pz])

                


class ParticleList():
    """List to store a collection of particle and being able to call their attricutes directly"""
    def __init__(self,nparticles,initial_condition,lim):
        self.nparticles = nparticles
        self.particles = np.array([Particle(mass,x,y,z, limit = lim) for x,y,z,mass in initial_condition])
        self.mass = np.array([self.particles[i].mass for i in range(len(self.particles))])
        self.position = np.array([self.particles[i].position for i in range(len(self.particles))])%lim
        self.momentum = np.array([self.particles[i].momentum for i in range(len(self.particles))])
        self.lim = lim

    def __len__(self):
        return self.nparticles

    def evolve(self,force, dt, boundary_periodic = True):
        for i in range(self.nparticles):
            self.position[i] += (self.momentum[i]*dt)
            if boundary_periodic == True:
                self.position[i] = self.position[i]%self.lim
            self.momentum[i] += (force*dt)
            #print(self.momentum[i])




class NBodySystem():
    """Descrribes the whole system of the nbody particles"""

    def __init__(self,nparticles, tfinal, mass=None, G=1, dim = 3, h = 1, lim = 10, boundary_periodic = True):
        self.lim = lim
        if boundary_periodic == False:
            self.lim_boundary = 2*lim
        else:
            self.lim_boundary = lim
        self.parameters = {}
        self.parameters['tfinal'] = tfinal
        #self.parameters['softener'] = softener
        self.parameters['G'] = G
        self.parameters['boundary_periodic'] = boundary_periodic
        self.parameters['nparticles'] = nparticles
        self.h = h
        self.dim = dim
        x0=np.random.randn(nparticles)%lim
        y0=np.random.randn(nparticles)%lim
        if dim == 3:
            z0=np.random.randn(nparticles)%lim
        else: 
            z0=np.zeros(nparticles)
        try:
            mass.any() == None
        except:
            mass = 2*np.ones(len(self))
        initial_condition = np.array([x0,y0,z0, mass]).transpose()
        self.particles = ParticleList(nparticles, initial_condition, lim)
        self.mesh = MeshGrid(lim=self.lim, dim=dim, h = h)
    def __len__(self):
        return self.parameters['nparticles']


    def compute_field(self):
        #lambda density r:np.array([self.particles.mass[i]*self.particles.position[i] if r = self.pos  for i in range(len(self))])
        density = self.mesh.weight_assesement(self.particles.position.transpose(), self.particles.mass)
        density = density.reshape([self.lim,self.lim,self.lim])
        #print(density)
        potential = np.zeros(len(self.mesh.tree.data))
        for i in range(len(potential)):
                potential[i] = np.array([self.parameters['G']/(np.linalg.norm(self.mesh.tree.data[i]))+0.1])
        potential = potential.reshape([self.lim,self.lim,self.lim])
        if self.lim != self.lim_boundary:
            density= np.pad(density, (0,self.lim_boundary-self.lim), 'constant', constant_values=0)
            potential = np.pad(potential, (0,self.lim_boundary-self.lim), 'constant', constant_values=0)
        phi = np.fft.irfft(np.fft.rfft(density.ravel())*np.fft.rfft(potential.ravel()))
        phi = phi.reshape([self.lim_boundary,self.lim_boundary,self.lim_boundary])
        return phi

    def grad_phi_mesh(self, phi,components):
        i = int(components[0]-0.5)
        j = int(components[1]-0.5)
        k = int(components[2]-0.5)
        lim = self.lim_boundary
        fx = -2*(phi[(i+1)%lim][j%lim][k%lim]-phi[(i-1)%lim][j%lim][k%lim])/(3*self.h)+(phi[(1+2)%lim][j%lim][k%lim]-phi[(i-2)%lim][j%lim][k%lim])/(12*self.h)
        fy = -2*(phi[i%lim][(j+1)%lim][k%lim]-phi[i%lim][(j-1)%lim][k%lim])/(3*self.h)+(phi[i%lim][(j+2)%lim][k%lim]-phi[i%lim][(j-2)%lim][k%lim])/(12*self.h)
        fz = -2*(phi[i%lim][j%lim][(k+1)%lim]-phi[i%lim][j%lim][(k-1)%lim])/(3*self.h)+(phi[i%lim][j%lim][(k+2)%lim]-phi[i%lim][j%lim][(k-2)%lim])/(12*self.h)
        #print(fx,fy,fz)
        return (fx,fy,fz)
    
    def compute_forces_mesh(self, phi):
        size = self.lim
        f = np.zeros([len(self.mesh.tree.data),3])
        for i in range(len(self.mesh.tree.data)):
            f[i] = self.grad_phi_mesh(phi,self.mesh.tree.data[i])
        return f
                     

    def evolve_system(self):
        phi = self.compute_field()
        force_m = self.compute_forces_mesh(phi)
        for i in range(len(self)):
            weight = self.mesh.weight_assesement(np.array([self.particles.position[i]]).transpose(), np.array([self.particles.mass[i]]))
            # print(force_m[0,int(ind[0]),int(ind[1]),int(ind[2])])
            fx = np.sum(force_m[:,0]*weight)
            fy = np.sum(force_m[:,1]*weight)
            fz = np.sum(force_m[:,2]*weight)
            forces = np.array([fx,fy,fz])
            print(forces)
            self.particles.evolve(forces,self.parameters['tfinal'], boundary_periodic = self.parameters['boundary_periodic'])


# system = NBodySystem(1 ,0.01)
# sys1 = system.particles.momentum
# system.evolve_system()



if __name__=='__main__':
    plt.ion()
    n=2
    oversamp=5
    system = NBodySystem(n,0.01, dim=2, boundary_periodic = True)
    #plt.plot(system.particles.position[:,0],system.particles.position[:,1],'*')
    plt.show()
    


    #fig = plt.figure()
    #ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    #line, = ax.plot([], [], '*', lw=2)

    for i in range(0,10000):
        for ii in range(oversamp):
            system.evolve_system()
        
        plt.clf()
        #print(system.particles.position)
        plt.plot(system.particles.position[:,0],system.particles.position[:,1],'*')
        plt.pause(1e-3)

