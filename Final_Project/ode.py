import numpy as np

def leap_frog(x,v,f_now,f_next,dt):
    """Update the particles position and momenta using the leap frog method
    -Arguments: - x     (array): The current position of the particles
                - v      (array): The current momenta of the particles
                - f_now  (array): The forces on the particles
                - f_next (array): The evolved forces on the particles
                - dt     (float): Time step to take to evolve
    -Retruns: -x_new (array): The new positions of the particles
              -v_new (array): The new momenta of the particles"""
    x_new = x+v*dt + 0.5*f_now*dt**2
    v_new = v+0.5*(f_now+f_next)*dt
    return x_new,v_new