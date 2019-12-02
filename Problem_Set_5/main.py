import numpy as np
from matplotlib import pyplot as plt
from util import *


n=1000
rad=100

f = open('Final_results.txt','w')

################Part 1#########################
#Initialize the problem
V, V_true, bc, mask, b,xx,yy = initial_settings(n,rad)
#Evolve the potential until it reaches the potential in V
V, count = evolve(V,bc,mask,b) 
#Compute the density of the charges
rho = compute_density(V)
#Computes the field for the potentials. Take only 10% of the point so that we don't only see arrows on the plots
V_sparse = V[::10,::10]
E = np.gradient(V_sparse)
V_true_sparse = V_true[::10,::10]
E_true = np.gradient(V_true_sparse)
#Create the plot
fig, ax = plt.subplots(1,3, figsize = (16,4))
ax0 = ax[0].pcolormesh(V, vmin=0,vmax=1)
ax[0].quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
fig.colorbar(ax0, ax=ax[0])
ax[0].set_title('Numerical Solution')
ax1 = ax[1].pcolormesh(V_true,vmin=0,vmax=1)
ax[1].quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E_true[1],-E_true[0])
ax[1].set_title('Analytic solution')
fig.colorbar(ax1, ax=ax[1])
ax2 = ax[2].pcolormesh(rho)
fig.colorbar(ax2, ax=ax[2])
ax[2].set_title('Rho')
fig.savefig('Part1_potential_relaxation.png')
#Compute lambda
rho = rho[np.abs(rho)>1e-10]
total_charge = np.sum(rho)
lam = total_charge/(2*np.pi*rad)

#Write the results into the summary file
f.write('-----------Part 1-------\n')
f.write(f'The relaxation method took {count} to converge to a precision of 0.05.\n')
f.write(f'The value of lambda on the ring is given by {lam}.\n')
f.write('We can see qualitatively in "Part1_potential_relaxation.png" that this is very close to the anlytic potential .\n')
f.write('However, since the analytic potential is a radial solution, it cannot meet the boundary condition of the box\n')
f.write('This explanins why the "analytic" field is really weird when we simply take the gradient of the field given by the analytic formula.\n')
f.write('Therefore, I would trust more the numerical solution since it actually solve for the good b.c\n')
f.write('Furthermore, our numerical charge denisty is exaclty what we would expect, ie the charge is on the outside of the cylinder.\n\n')


################Part 2#########################
#Reinitialize V to the initial one so that we can compare
V, V_true, bc, mask, b,xx,yy = initial_settings(n,rad)
#Evolve the potenital unsing conjugate gradient until it reaches a certain tolerance treshold
V, count = evolve_conjgrad(V,mask,b)
#Compute the charge density
rho = compute_density(V)
#Compute the electric field
V_sparse = V[::10,::10]
E = np.gradient(V_sparse)
V_true_sparse = V_true[::10,::10]
E_true = np.gradient(V_true_sparse)
#Create plots
fig, ax = plt.subplots(1,3, figsize=(16,4))
ax0 = ax[0].pcolormesh(V, vmin=0,vmax=1)
ax[0].quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
fig.colorbar(ax0, ax=ax[0] )
ax[0].set_title('Numerical Solution')
ax1 = ax[1].pcolormesh(V_true,vmin=0,vmax=1)
ax[1].quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E_true[1],-E_true[0])
fig.colorbar(ax1, ax=ax[1])
ax[1].set_title('Analytic solution')
ax2 = ax[2].pcolormesh(rho)
fig.colorbar(ax2, ax=ax[2])
ax[2].set_title('Rho')
fig.savefig('Part2_Potential_conjugate_gradient.png')
#Write the results into the summary file
f.write('-----------Part 2-------\n')
f.write(f'The relaxation method took {count} to converge to a precision of 0.05, which is way faster than what we had in one.\n')
f.write('Resulting potential,field and chage distribution are shown in "Part2_Potential_conjugate_gradient.png".\n\n')


##############Part 3##################
#Start with lower resolution
n_lowres=200
rad_lowres=20
#Initialize the potential at this lower resolution
V, V_true, bc, mask, b,_,_ = initial_settings(n_lowres,rad_lowres)
#Evolve the potential by updating the resolution
V,xx,yy,count = evolve_resolution(V,mask,b,n,rad_lowres,n_lowres)
#Compute the Electric-fields
V_sparse = V[::10,::10]
E = np.gradient(V_sparse)
#Plots the result
plt.pcolormesh(V, vmin=0,vmax=1)
plt.colorbar()
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
plt.savefig('Part3_field_resolution_increase.png')
#Write the results into the summary file
f.write('-----------Part 3-------\n')
f.write(f'The relaxation method took {count} to converge to a precision of 0.05, which is less than in Part 2.\n')
f.write('The interpolation for scipy is relatively slow for large matrices so I am not sure if doing this this way\n')
f.write('actually solves time... This could  although definitely be address and then it would save much time.\n')
f.write('Resulting potential,field and chage distribution are shown in "Part3_field_resolution_increase.png".\n\n')


#############Part 4##################
#Reinitialize the potential at lower resolution
V, V_true, bc, mask, b, xx, yy = initial_settings(n_lowres,rad_lowres)
#Adds a bump of 10% of the diameter of the wire to the boundary condition
condition = (xx-n_lowres//2-rad_lowres)**2 + (yy-n_lowres//2)**2 <= (2*rad_lowres//10)**2
mask[condition] = True 
bc[condition] = 1
V[mask]=bc[mask]
#Evolve the potential by updating the resolution
V,xx,yy,count = evolve_resolution(V,mask,b,1000,rad_lowres,n_lowres)
#Computes the E-field
V_sparse = V[::10,::10]
E = np.gradient(V_sparse)
#Plots the results
plt.pcolormesh(V,vmin=0,vmax=1)
plt.colorbar()
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
plt.savefig('Part4_Field_from_wire_with_bump.png')
#Write the results into the summary file
f.write('-----------Part 4-------\n')
f.write('The field near the bump is enormous compared to the rest of the field as it can be see in "Part4_Field_from_wire_with_bump.png".\n')
f.write('Therefore, a  lot of power is radiated away because of the bump.\n')
f.write('This explains why power companies do not want bumps in their wires.\n\n')



##############Part 5##################
#Constants to solve the PDEs
dt = 0.0005
dx = 0.0005
k = 10**(-4)
x_max = 0.01
t_max = 5
C = 100
#Solve the PDEs
x,T = evolve_von_neumann_rod(dt,dx,t_max,x_max,k,C)
#Create the plot
plot_times = np.arange(0,1.0,0.1)
for t in plot_times:
    index = int(t/dt)
    plt.plot(x,T[index,:], label = f't={t}')
plt.xlabel('Position x')
plt.ylabel('Temperature')
plt.legend()    
plt.savefig('Part5_Heat_equation.png')
#Write the results into the summary file
f.write('-----------Part 5-------\n')
f.write('Note that since we are only considering a slab in the middle and that all slab are the same, I solve the problem in 1D.\n')
f.write('We have to define the following constants: dt,dx,k,x_max,t_max.\n')
f.write('While x_max,t_max and C are simply a choice or would be fix by the problem,\n')
f.write('the value of dt,dx and k decide of the stability of the solution. If k*dt/dx**2<1, the solution will converge.\n\n')
f.close()

