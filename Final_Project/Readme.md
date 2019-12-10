N-Body Simulation
=================

Part 1
------
We can see in [Part1.gif](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part1.gif) that the particle stays motionless when using my code. Furthermore, [Part1_acceleration.txt](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part1_acceleration.txt) shows that the particles feels no force since its acceleration remains 0.

Part 2
------
We can see in [Part2.gif](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part2.gif) that the two particles remain in orbit. Note
that the initial condition for the orbit have been found by trial and error, and we thus not start with a perfect orbit, thus the change in the radius of said orbit.


Part 3
------

We can see in [Part3_non_periodic_soft=0.8_dt=1_n=300000_size=500.gif](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part3_non_periodic_soft=0.8_dt=1_n=300000_size=500.gif) that for non-periodic condition, mass rapidely collapse into one point at the initial point of higher density 
even if there are some formation of smaller structure during the process. Furthermore, after they collapse, the blob "explodes" since we have a softening and
therefore the particles do not feel a force when they are to close, and just pass right through each other at very high speed. Furhtermore, since we can see in 
[Part3_non_periodic.txt](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part3_non_periodic.txt) that energy is not conserved. Although this
might seem worrying at first glance, this is to be expected since the codes deletes partciles that leaves the grid in that settings, which of course changes the energy
of the system.

For the case of the periodic conditions, we see that a lot of smaller structure (that ressemble stars or galaxies) are being formed in [Part3_periodic_soft=0.8_dt=1_n=300000_size=500.gif](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part3_periodic_soft=0.8_dt=1_n=300000_size=500.gif) before disolving
due to the softening once again. In this case, energy is conserved as expected since all particle are conserved (boundary condtion done in a way that the problem
is solved on a toroïdal surface), as we can see in [Part3_periodic.txt](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part3_periodic.txt) 


Part 4
------

When assigning an initial mass density of k^-3, we get more stable structure formed then when starting with a uniform distribution. This is because some places
on the grid already have a higher density at those point. This can be seen in [Part4_periodic_soft=10_dt=100_n=300000_size=500.gif](https://github.com/antoinebelley/Phys512_assignments/blob/master/Final_Project/Part4_periodic_soft%3D10_dt%3D100_n%3D300000_size%3D500_density-k%5E-3.gif) which shows the density in logscale.
