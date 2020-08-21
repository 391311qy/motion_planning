'''Boor's algorithm to generate b spline'''
'''https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/de-Boor.html'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import proj3d

class B_spline:

    def __init__(self):
        self.U = []

    def de_boor(self, u):
        '''
        -----
        params: a value u
        output: the point on the curve, p(u)
        '''
        pass

    def N(self, i, p , u):
        # basis function
        if p == 0:
            if self.U[i] <= u < self.U[i+1]:
                return 1
            else: return 0
        return (u - self.U[i]) / (self.U[i+p] - self.U[i]) * self.N(i, p-1, u) + \
                (self.U[i+p+1] - u) / (self.U[i+p+1] - self.U[i+1]) * self.N(i+1, p-1, u)


# def pad_knot(knots, p):
#     t = []
#     for i in knots:
#         t.append([i[0]]*p + list(i) + [i[-1]]*p)
#     return t

# knots = np.array([[1,2,3],[3,4,5],[2,5,2]])


# ax = plt.subplot(111, projection='3d')
# ax.scatter3D(knots[:, 0],knots[:, 1],knots[:, 2])

# plt.show()
