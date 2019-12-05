import matplotlib.pyplot as plt
import matplotlib.animation as animation
from particle import *



n=int(3e5)
size=500
soft =8
dt = 100

##################Plot for the non-periodic boundary condition##########################
f = open('Part3_energy_non_periodic.txt', 'w')
system = NBodySystem(n,size, mass=1/n, boundary_periodic = False, softner = soft)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))


ptcl = ax.imshow(system.grid_pos,origin='lower', vmin=system.grid_pos.min(),vmax=system.grid_pos.max(),cmap=plt.get_cmap('inferno'))
plt.colorbar(ptcl)

def animate(i):
    """perform animation step"""
    global system, ax, fig, dt, f
    system.evolve_system(dt,energy_file=f)
    ptcl.set_data(system.grid_pos)
    ptcl.set_clim(system.grid_pos.min(),system.grid_pos.max())
    ptcl.set_cmap(plt.get_cmap('inferno'))
    return ptcl,




ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=10)

ani.save(f'Part3_non_periodic_soft={soft}_dt={dt}_n={n}_size={size}.gif', writer='imagemagick')
f.close()

####################Plot for the periodic boundary condition##########################
n=int(3e5)
size=500
soft =3
dt = 30


f = open('Part3_energy_periodic.txt', 'w')
system = NBodySystem(n,size, mass=1/n, boundary_periodic = True, softner = soft)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))


ptcl = ax.imshow(system.grid_pos,origin='lower', vmin=system.grid_pos.min(),vmax=system.grid_pos.max(),cmap=plt.get_cmap('inferno'))
plt.colorbar(ptcl)

def animate(i):
    """perform animation step"""
    global system, ax, fig, dt, f
    system.evolve_system(dt, energy_file =f)
    ptcl.set_data(system.grid_pos)
    ptcl.set_clim(system.grid_pos.min(),system.grid_pos.max())
    ptcl.set_cmap(plt.get_cmap('inferno'))
    return ptcl,


ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=10)

ani.save(f'Part3_periodic_soft={soft}_dt={dt}_n={n}_size={size}.gif', writer='imagemagick')
f.close()