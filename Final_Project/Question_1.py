import matplotlib.pyplot as plt
import matplotlib.animation as animation
from particle import *

n=int(1)
size=40
system = NBodySystem(n,size, mass=1, boundary_periodic = True)
dt=1

f = open("Part1_acceleration.txt","w")
f.write('a_x a_y\n')

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))
f.write(f'{system.acc[0,0]} {system.acc[0,1]}\n')

ptcl, = ax.plot([], [], '*')

def animate(i):
    """perform animation step"""
    global system, ax, fig, dt
    system.evolve_system(dt)
    ptcl.set_data(system.particles.position[:,0],system.particles.position[:,1])
    f.write(f'{system.acc[0,0]} {system.acc[0,1]}\n')
    return ptcl,



ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=10)
ani.save('Part1.gif', writer='imagemagick')

f.close()