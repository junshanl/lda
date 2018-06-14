import numpy as np

print np.outer([[1,1],[1,1]],[[1,2],[3,4]])

a = np.array([[1,2],[3,4]])
a[:, [0,1]]+=2

print a
