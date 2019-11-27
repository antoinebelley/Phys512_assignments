import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

def initial_settings(n,r):
    """Initialize the problem by computing the potential and boundary condition for a certain resolution
    and radius.
    -Arguments: -n        (int): Resolution of the grid
                -r        (int): Radius of the wire
    -Returns:   -V      (array): The initial potential on the grid 
                -true_V (array): The analytic solution of the potential. Note that this solution cannot fully
                                 solve the boundary condition so it is approximated.
                -bc    (array) : Boundary condition of the problem
                -mask  (array) : Mask for where to apply the boundary condition
                -b     (array) : 'Solution' vector from the boundary condition
                -xx    (array) : x-coord of the grid
                -yy    (array) : y-coord of the grid"""
    #Create grid to help define the circle
    size = np.linspace(0,n,n)
    xx,yy = np.meshgrid(size,size)
    #Initialize th potential and bc arrays
    V=np.zeros(xx.shape)
    bc=0*V
    #Condition that defines the circle
    cx, cy, r = n//2, n//2, r
    condition = (xx-cx)**2 + (yy-cy)**2 <= r**2
    #Creates the mask and boundary condition
    mask = condition.copy()
    mask[:,0]=True
    mask[:,-1]=True
    mask[0,:]=True
    mask[-1,:]=True
    mask[condition]=True
    bc[condition]=1
    #Compute the real potential. Finds the charge density by computing the field for lambda =1
    #And then computes the slope necessary to solve the boundary condition
    X = xx.ravel()-cx
    Y = yy.ravel()-cy
    norm = np.sqrt(X**2+Y**2)-r
    true_V = np.log(norm)
    true_V = true_V.reshape(xx.shape)
    temp_V = true_V
    slope=1/temp_V[cx,0]
    true_V = -slope*true_V+1
    true_V[condition] = 1
    #Comutes the 'solution' vector from the boundary condition
    b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
    return V, true_V, bc, mask, b,xx,yy


def take_one_step(V,bc,mask):
    """Evolve the potential with the relaxation technique by one step
    -Arguments: - V    (array): The potential on the grid
                - bc   (array): Boundary condition of the problem
                - mask (array): Mask for where to apply the boundary condition
    -Returns:   - V    (array): The evolve potential by on step"""
    V[1:-1,1:-1]=(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
    V[mask]=bc[mask]
    return V

def compute_density(V):
    """Computes the charge density assiociated with this potential
    -Arguments: - V   (array): The potential on the grid
    -Returns  : - rho (array): The chrage density on the grid"""
    return V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0


def evolve(V ,bc, mask,b, tol_V=5e-2, lim=10000):
    """Evolve the potential with the relaxation method until it matches the tolerance level or reaches the limit
    -Arguments: - V     (array) : The potential on the grid
                - bc    (array) : Boundary condition of the problem
                - mask  (array) : Mask for where to apply the boundary condition
                - b     (array) : 'Solution' vector from the boundary condition
                - tol_V   (int) : The tolerance level that we hope to achieve. Default = 5e-2
                - lim     (int) : The limit for the iteration. Default = 1000000
    -Returns  : - V     (array) : The evolved potential on the grid
                - count   (int) : The number of steps to converge"""
    count= 0
    r=b-Ax(V,mask)
    for i in range(lim):
        take_one_step(V,bc,mask)
        rtr=np.sum(r*r)
        r=b-Ax(V,mask)
        count += 1
        if rtr>tol_V: continue
        else: break
    return V, count

def evolve_conjgrad(V, mask,b,tol_V=5e-2,lim=10000):
    """Evolve the potential whith the conjugate gradient until it matches the tolerance level or reaches the limit
    -Arguments: - V     (array) : The potential on the grid
                - mask  (array) : Mask for where to apply the boundary condition
                - b     (array) : 'Solution' vector from the boundary condition
                - tol_V   (int) : The tolerance level that we hope to achieve. Default = 5e-2
                - lim     (int) : The limit for the iteration. Default = 1000000
    -Returns  : - V     (array) : The evolved potential on the grid
                - count   (int) : The number of steps to converge"""
    r=b-Ax(V,mask)
    p=r.copy()
    count = 0
    for k in range(lim):
        Ap=(Ax(pad(p),mask))
        rtr=np.sum(r*r)
        print(rtr)
        if rtr < tol_V:
            break
        count += 1
        alpha=rtr/np.sum(Ap*p)
        V=V+pad(alpha*p)
        rnew=r-alpha*Ap
        beta=np.sum(rnew*rnew)/rtr
        p=rnew+beta*p
        r=rnew
    return V, count



def Ax(V,mask):
    """The vector that we are searching times the potential formula in matrix form
    -Arguments: V  (array) : The potential on the grid
    -Returns  : Ax (array) : The evolved potential vectors"""
    Vuse=V.copy()
    Vuse[mask]=0
    ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
    ans=ans-V[1:-1,1:-1]
    return ans

def pad(A):
    """Pads an array with a layers of zero all around:
    -Arguments: -A (array): The array to be padded
    -Retunrns:  -A (array): The padded array"""
    AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
    AA[1:-1,1:-1]=A
    return AA

def grid_interpolate(r,n,resolution, V):
    """Interpolate the potential on the grid to a higher resolution grid
       -Arguments: - r          (int) : The radius of the circle on the grid
                   - n          (int) : The size of the current grid (lower resolution)
                   - resoltuion (int) : The resolution that we want to achieve
                   - V        (array) : The potential on the grid of lower resolution
       -Returns:   - V _new   (array) : The interpolated potential on the higher resolution grid
                   - mask     (array) : Mask for where to apply the boundary condition on higher resolution grid
                   - bc       (array) : Boundary condition of the problem on higher resolution grid
                   """
    size = np.linspace(0,n,n)
    xx_previous,yy_previous = np.meshgrid(size,size)
    size = np.linspace(0,n,resolution)
    xx,yy = np.meshgrid(size,size)
    points = np.array([xx_previous.ravel(),yy_previous.ravel()]).transpose()
    V = V.ravel() 
    bc=np.zeros(xx.shape)
    cx, cy, r = n//2, n//2, r
    condition = (xx-cx)**2 + (yy-cy)**2 <= r**2
    mask = condition.copy()
    mask[:,0]=True
    mask[:,-1]=True
    mask[0,:]=True
    mask[-1,:]=True
    mask[condition]=True
    bc[condition]=1
    V_new = interp.griddata(points,V,(xx,yy))
    return V_new, mask, bc


def evolve_resolution(V,mask,b,resolution,r,n,tol_V=5e-2,lim=10000, steps=200):
    """Interpolate the potential on the grid to a higher resolution grid
       -Arguments: - V        (array) : The initial potential on the grid of lower resolution
                   - mask     (array) : Mask for where to apply the boundary condition
                   - b        (array) : 'Solution' vector from the boundary condition
                   - resoltuion (int) : The resolution that we want to achieve
                   - r          (int) : The radius of the circle on the grid
                   - n          (int) : The size of the current grid (lower resolution)
                   - tol_V      (int) : The tolerance level that we hope to achieve. Default = 5e-2
                   - lim        (int) : The limit for the iteration. Default = 1000000
                   - steps      (int) : Size of the steps bewtween the reosultion sizes. Default = 200            
       -Returns:   - V        (array) : The interpolated potential on the higher resolution grid
                   - bc       (array) : Boundary condition of the problem on higher resolution grid
                   - xx       (array) : The x-coord of the grid of higher resolution
                   - yy       (array) : The y-coord of the grid of higher resolution
                   - count      (int) : The number of steps to converge"""

    count = 0
    V,i = evolve_conjgrad(V, mask, b, tol_V=tol_V, lim=lim)
    count += i
    while n < resolution:
        if n+steps<resolution: n_new = n+steps
        else:  n_new = resolution
        V, mask, bc = grid_interpolate(r,n,n_new,V)
        b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
        V, i = evolve_conjgrad(V, mask, b, tol_V=tol_V, lim=lim)
        n=n_new 
        count +=i
    size = np.linspace(0,resolution,resolution)
    xx,yy = np.meshgrid(size,size)
    return V,xx,yy, count



def evolve_von_neumann_rod(dt,dx,t_max,x_max,k,C):
    """Solve the heat equation using the Forward-Time-Central-Space method for von Neumann boundary condition
    u_x(0,t) = C whee C is a constnant.
    -Arguments: - dt    (float): The size of the time steps
                - dx    (float): The size of the spatial steps
                - t_max (float): The maximal time to which we solve the pdes
                - x_max (float): The maximal size of the space on which we solve the pdes
                - k     (float): Coefficient to assure convergence. We need that k*dt/dx**2 <= 0.5 for the solution to be stable
                - C     (float): The rate for the Von Neumann condition 
    -Returns:   - x     (array): The position on the rod
                - T     (array): The temperature on the rod as a function of time and position"""
    #Factor that decides convergence
    s = k*dt/dx**2
    if s > 1/2:
        print('ERROR: s must be under 1/2 for the pdes to converge.')
        print(f'Current value of s is {s}')
        print('Exiting...')
        exit(1)
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    T = np.zeros([r,c])
    for n in range(0,r-1):
        for j in range(1,c-1):
            #Von Neumann Boundary Condition
            T[n,0] = t[n]*C
            #Solve the PDE
            T[n+1,j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j+1]) 
    return x,T