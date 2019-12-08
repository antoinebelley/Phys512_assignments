import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from particle import *



n = int(3e5)
size=500
soft =0.8
dt = 1


system = NBodySystem(n,size, mass=1/n, boundary_periodic = True,  early_universe = True, softner = soft)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))
min_lim = system.grid_pos[system.grid_pos>0].min()
max_lim = system.grid_pos.max()
sys = system.grid_pos.copy()
sys[sys==0] = min_lim*1e-3
# ptcl = ax.imshow(sys, norm=LogNorm(vmin=min_lim, vmax=max_lim),origin='lower', cmap=plt.get_cmap('inferno'))
ptcl = ax.imshow(system.grid_pos, vmin=system.grid_pos.min(), vmax=system.grid_pos.max(),origin='lower', cmap=plt.get_cmap('inferno'))
plt.colorbar(ptcl)




def animate(i):
    """perform animation step"""
    global system, ax, fig, dt, f
    system.evolve_system(dt)
    min_lim = system.grid_pos[system.grid_pos>0].min()
    max_lim = system.grid_pos.max()
    sys = system.grid_pos.copy()
    sys[sys==0] = min_lim*1e-3
    # ptcl.set_data(sys)
    # ptcl.set_norm(LogNorm(vmin=min_lim, vmax=max_lim))
    ptcl.set_data(system.grid_pos)
    ptcl.set_clim(vmin=system.grid_pos.min(), vmax=system.grid_pos.max())
    ptcl.set_cmap(plt.get_cmap('inferno'))
    return ptcl,




ani = animation.FuncAnimation(fig, animate, frames=3000,
                              interval=10)

# plt.show()
ani.save(f'Part4_periodic_soft={soft}_dt={dt}_n={n}_size={size}_density-k^-3.gif', writer='imagemagick')
