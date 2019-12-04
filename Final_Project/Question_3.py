import matplotlib.pyplot as plt
import matplotlib.animation as animation
from particle import *



n=int(5e5)
size=400
soft =10
dt = 1

####################Plot for the non-periodic boundary condition##########################
# f = open('Part3_energy_non_periodic.txt', 'w')
# system = NBodySystem(n,size, mass=1/n, boundary_periodic = False, softner = soft)


# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))


# ptcl = ax.imshow(system.grid)


# def animate(i):
#     """perform animation step"""
#     global system, ax, fig, dt, f
#     system.evolve_system(dt,energy_file=f)
#     ptcl.set_data(system.grid)
#     return ptcl,




# ani = animation.FuncAnimation(fig, animate, frames=500,
#                               interval=10)
# ani.save('Part3_non_peridoic.gif', writer='imagemagick')
# f.close()

####################Plot for the periodic boundary condition##########################
f = open('Part3_energy_periodic.txt', 'w')
system = NBodySystem(n,size, mass=1/n, boundary_periodic = True, softner = soft)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))


ptcl = ax.imshow(system.grid)

def animate(i):
    """perform animation step"""
    global system, ax, fig, dt, f
    system.evolve_system(dt, energy_file =f)
    ptcl.set_data(system.grid)
    return ptcl,


ani = animation.FuncAnimation(fig, animate, frames=500,
                              interval=10)

ani.save('Part3_periodic2.gif', writer='imagemagick')
f.close()