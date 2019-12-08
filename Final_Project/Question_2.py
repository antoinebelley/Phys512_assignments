import matplotlib.pyplot as plt
import matplotlib.animation as animation
from particle import *

n=int(2)
size=300
pos = np.array([[size//2+0.01-10,size//2+0.01],[size//2+10.01,size//2+0.01]])
mom = np.array([[0,0.17],[0,-0.17]])
system = NBodySystem(n,size, mass=5, boundary_periodic =False, position=pos,momentum=mom)
dt=10

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))
#ax.grid()

ptcl, = ax.plot([], [], '*')


def animate(i):
    """perform animation step"""
    global system, ax, fig, dt
    system.evolve_system(dt)
    ptcl.set_data(system.particles.position[:,0],system.particles.position[:,1])
    #ptcl.set_markersize(ms)
    return ptcl,

ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=10)

ani.save('Part2.gif', writer='imagemagick')