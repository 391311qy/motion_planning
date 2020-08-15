'''
@ author: yue qi
lqRRTstar for optimal kinodynamic motion planning, possibly using ROS in future

source:
Perez, Alejandro, et al. 
"LQR-RRT*: Optimal sampling-based motion planning with automatically derived extension heuristics." 
2012 IEEE International Conference on Robotics and Automation. IEEE, 2012.
'''

import numpy as np
import numpy.linalg as npl
from numpy.matlib import repmat

from env3D import env
from LQR import lqr


class lqrrt:

    def __init__(self):
        self.env = env()
        self.V = set() # set of positions
        self.E = set() # set of edges
        self.X = set() # set of states
        self.Parent = {}

##################################### LQR RRT star implementation    
    def LQR_rrt_star(self):
        xrand = self.sampleFree()
        xnearest = self.LQRNearest(self.V, xrand)
        xnew = self.LQRSteer(xnearest, xrand)
        Xnear = self.LQRNear(self.V, xnew)
        xmin, sig_min = self.ChooseParent(Xnear, xnew)
        if self.collisionFree(sig_min):
            self.X.add(xnew)
            self.E.add((xmin, xnew))
            self.V, self.E = self.rewire(self.V, self.E, Xnear, xnew)

    def ChooseParent(self, Xnear, xrand):
        minCost = np.inf
        xmin = None
        sig_min = None
        for xnear in Xnear:
            sigma = self.LQRsteer(xnear, xrand)
            newcost = self.cost(xnear) + self.cost(sigma)
            if newcost < minCost:
                minCost, xmin = newcost, xnear
                sig_min = sigma
        return xmin, sig_min

    def rewire(self, V, E, Xnear, xnew):
        for xnear in Xnear:
            sig = self.LQRsteer(xnew, xnear)
            if self.cost(xnew) + self.cost(sig) < self.cost(xnear):
                if self.collisionFree(sig):
                    xparent = self.Parent[xnear]
                    E.remove((xparent, xnear))
                    E.add((xnew, xnear))
        return V, E

    def LQRNear(self, V, x):
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        diff = (V - x) # 10 * 3
        near = diff.T@S@diff
        xmin = V[min(near)]

    