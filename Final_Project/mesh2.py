import numpy as np
from scipy import spatial

h=0.5
lim=10


x,y,z = np.mgrid[0.5*h:lim-0.5*h+1:h, 0.5*h:lim-0.5*h+1:h, 0.5*h:lim-0.5*h+1:h]
r = np.array([0,1,2])
tree = spatial.KDTree(list(zip(x.ravel(), y.ravel(), z.ravel())))
pts = np.array([[0, 0], [2.1, 2.9]])
d,i = tree.query(r)
