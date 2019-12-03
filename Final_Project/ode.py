import numpy as np

def leap_frog(x,v,f_now,f_next,dt):
    x_new = x+v*dt + 0.5*f_now*dt**2
    v_new = v+0.5*(f_now+f_next)*dt
    return x_new,v_new