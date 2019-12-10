import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from particle import *



n = int(3e5)
size=500
soft =10
dt = 100


system = NBodySystem(n,size, mass=100, boundary_periodic = True,  early_universe = True, softner = soft)

grid = system.grid_pos.copy()
grid[grid == 0] = grid[grid!=0].min()*1e-3
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim = (0,size),ylim=(0,size))
ptcl = ax.imshow(system.grid_pos, norm = LogNorm(vmin = grid.min(), vmax = grid.max()),origin='lower', cmap=plt.get_cmap('inferno'))
plt.colorbar(ptcl)



count=0
def animate(i):
    """perform animation step"""
    global count
    print(count)
    global system, ax, fig, dt, f
    for i in range(5):
        system.evolve_system(dt)
    grid = system.grid_pos.copy()
    grid[grid == 0] = grid[grid!=0].min()*1e-3
    ptcl.set_data(system.grid_pos)
    ptcl.set_norm(LogNorm(vmin = grid.min(), vmax = grid.max()))
    ptcl.set_cmap(plt.get_cmap('inferno'))
    count+=1
    return ptcl,




ani = animation.FuncAnimation(fig, animate, frames=100,
                              interval=10)

# plt.show()
ani.save(f'Part4_periodic_soft={soft}_dt={dt}_n={n}_size={size}_density-k^-3.gif', writer='imagemagick')
