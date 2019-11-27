import numpy as np
from scipy import spatial
from numba import njit

class mesh():
    def __init__(self, h, boundary_condition):
        self.step = 1/h
        self.h = h
        if boundary_condition==True:
            size = np.linspace(-1,1,self.step+1)
        else:
            size = np.linspace(0,2,2*self.step+2)
        self.x,self.y  = np.meshgrid(size,size)
        #print(self.x)
        self.raveled = list(zip(self.x.ravel(), self.y.ravel()))
        #print(self.raveled)
        self.tree = spatial.KDTree(self.raveled)

    def density(self, r, mass):
        r = r.transpose()
        index = self.tree.query(r)[1]
        density = np.zeros(self.x.shape)
        indices = np.zeros([len(r),2])
        for j in range(len(index)):
            i = index[j]
            ix = np.argwhere(self.x==self.raveled[i][0])[0][1]
            iy = np.argwhere(self.y==self.raveled[i][1])[1][0]
            indices[j]=[int(ix),int(iy)]
            density[ix][iy] += mass[j]

        return density,indices

