import numpy as np
import time
import matplotlib.pyplot as plt
from mesh import mesh

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

    def evolve(self,force, dt, boundary_periodic = True):
        for i in range(self.nparticles):
            self.position[i] += (self.momentum[i]*dt)
            self.momentum[i] += (force[i]*dt)
            print(f'force: {force[i]}')
            print(f'position: {self.position[i]}')
            print(f'momentum: {self.momentum[i]}')
            




class NBodySystem():
    """Descrribes the whole system of the nbody particles"""

    def __init__(self,nparticles, tfinal, mass=None, G=1, h = 0.01, boundary_periodic = True, softner=0.001):
        self.parameters = {}
        self.parameters['tfinal'] = tfinal
        self.parameters['softener'] = softner
        self.parameters['G'] = G
        self.parameters['boundary_periodic'] = boundary_periodic
        self.parameters['nparticles'] = nparticles
        self.h = h
        x0=np.random.rand(nparticles)
        y0=np.random.rand(nparticles)
        exp_x0=np.random.randint(2, size=nparticles)
        exp_y0=np.random.randint(2, size=nparticles)
        #x0 = x0*np.power(-1,exp_x0)
        #y0 = y0*np.power(-1,exp_y0)
        try:
            mass.any() == None
        except:
            mass = 0.01/len(self)*np.ones(len(self))
        initial_condition = np.array([x0,y0, mass]).transpose()
        self.particles = ParticleList(nparticles, initial_condition)
        self.mesh = mesh(self.h, boundary_condition=boundary_periodic)
    def __len__(self):
        return self.parameters['nparticles']


    def compute_field(self):
        step = int(1/self.h +1)
        density, self.index = self.mesh.density(self.particles.position.transpose(), self.particles.mass)
        self.density = density
        green = np.zeros(self.mesh.x.size)
        for j in range(len(self)):
            for i in range(len(green)):
                rsqr = np.linalg.norm(self.mesh.raveled[i])
                soft = self.parameters['softener']**2
                if rsqr<soft**2: rsqr=soft
                rsqr=rsqr+self.parameters['softener']**2
                green[i] += np.array([self.parameters['G']/rsqr])
        green = green.reshape(self.mesh.x.shape)
        green = green.transpose()
        if self.parameters['boundary_periodic'] == False:
            #green[:step:, :step] = np.fft.fftshift(green[:step:, :step])
            # plt.pcolormesh(self.mesh.x,self.mesh.y,green)
            # plt.colorbar()
            # plt.show()
            green[step:, :step] = green[:step:, :step]
            green[:step, step:] = green[:step, :step:]
            green[step:, step:] = green[:step:, :step:]
        phi = np.abs(np.fft.ifft2(np.fft.fft2(density)*np.fft.fft2(green)))
        phi = np.fft.ifftshift(phi)
        # plt.pcolormesh(self.mesh.x,self.mesh.y,density)
        # plt.colorbar()
        # plt.show()
        plt.pcolormesh(self.mesh.x,self.mesh.y,green)
        plt.colorbar()
        plt.show()
        # plt.pcolormesh(self.mesh.x,self.mesh.y,phi)
        # plt.colorbar()
        # plt.show()

        
        return phi

    def grad_phi_mesh(self, phi,components,step):
        i = int(components[0])
        j = int(components[1])
        if self.parameters['boundary_periodic'] == True:
            fx = 2*(phi[(i+1)%step][j]-phi[(i-1)][j])/(3*self.h)-(phi[(i+2)%step][j]-phi[(i-2)][j])/(12*self.h)
            fy = 2*(phi[i][(j+1)%step]-phi[i][(j-1)])/(3*self.h)-(phi[i][(j+2)%step]-phi[i][(j-2)])/(12*self.h)
        else:
            try:
                fx = 2*(phi[(i+1)][j]-phi[(i-1)][j])/(3*self.h)-(phi[(i+2)][j]-phi[(i-2)][j])/(12*self.h)
            except:
                fx = 0
            try:
                fy = 2*(phi[i][(j+1)]-phi[i][(j-1)])/(3*self.h)-(phi[i][(j+2)]-phi[i][(j-2)])/(12*self.h)
            except:
                fy = 0

        return (fx,fy)
    
    def compute_forces_mesh(self, phi):
        if self.parameters['boundary_periodic']==True:
            step = int(1/self.h +1)
        else:
            step = int(2/self.h +2)
        f = np.zeros([step,step,2])
        for i in range(0,step):
            for j in range(0,step):
                f[i][j] = self.grad_phi_mesh(phi,(i,j), step)
        return f
                     

    def evolve_system(self):
        phi = self.compute_field()
        force_m = self.compute_forces_mesh(phi)
        forces = np.zeros([len(self),2])
        for i in range(len(self)):
            x,y = self.index[i]
            x = int(x)
            y = int(y)
            forces[i][0] = force_m[x,y,0]
            forces[i][1] = force_m[x,y,1]
        # plt.pcolormesh(self.mesh.x, self.mesh.y,force_m[:,:,0])
        # plt.colorbar()
        # plt.show()
        # plt.pcolormesh(self.mesh.x, self.mesh.y,force_m[:,:,1])
        # plt.colorbar()
        # plt.show()
        self.particles.evolve(forces,self.parameters['tfinal'], boundary_periodic = self.parameters['boundary_periodic'])



if __name__=='__main__':
    plt.ion()
    n=2
    system = NBodySystem(n,0.1, boundary_periodic = True)
    system.evolve_system()
    # system.evolve_system()
    # system.evolve_system()
    plt.plot(system.particles.position[:,0],system.particles.position[:,1],'*')
    plt.show()

    for i in range(0,100):
        #for ii in range(5):
        system.evolve_system()
        
        plt.clf()
        plt.plot(system.particles.position[:,0],system.particles.position[:,1],'*')
        #print(system.particles.position[:,0],system.particles.position[:,1])
        plt.pause(1e-3)

