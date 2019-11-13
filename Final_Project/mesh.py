import numpy as np
from scipy import spatial

class MeshGrid():
    def __init__(self, lim = 10, dim =2, h = 1):
        self.h = h
        self.lim = lim
        self.dim = dim
        x,y,z = np.mgrid[0.5*h:lim-0.5*h+1:h, 0.5*h:lim-0.5*h+1:h, 0.5*h:lim-0.5*h+1:h]
        self.tree = spatial.KDTree(list(zip(x.ravel(), y.ravel(), z.ravel())))

    def weight_assesement(self, r, mass):
        heavyside = np.zeros(len(self.tree.data))
        r = r.transpose()
        d,index = self.tree.query(r)
        for i in range(len(index)):
            heavyside[index[i]] += mass[i]/(self.h**3)
        return heavyside

