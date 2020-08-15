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

from inv_pendulum import env
from LQR import lqr


class lqrrt:

    def __init__(self):
        self.env = env()
        self.V = set() # set of positions
        self.E = set() # set of edges
        self.X = set() # set of states
        self.Parent = {}
        self.gamma = 1
        self.LQR = lqr()
        self.COST = {}

    def LinearizedMotionModel(self, x, u):
        # x: tuple state
        # u: array contorl policy
        A, B = self.LQR.Linearized_Motion_Model(x, u)
        x = np.array(x)
        u = np.array(u)
        return tuple(A@x + B@u)
        # define A(x0,u0), B(x0,u0), Q, R here

    def getDist(self, x1, x2):
        # determined by the model and environemnt, assume work in 3D env
        return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2)

    def collisionFree(self, x1, x2):
        pos1, pos2 = x1[0:3], x2[0:3] # still tuple
        collide, dist = self.env.isCollide(pos1, pos2)
        return collide, dist

##################################### LQR RRT star implementation    
    def LQR_rrt_star(self):
        xrand = self.env.sampleFree()
        xnearest = self.LQRNearest(self.V, xrand)
        xnew = self.LQRsteer(xnearest, xrand)
        Xnear = self.LQRNear(self.V, xnew)
        xmin, sig_min = self.ChooseParent(Xnear, xnew)
        # TODO: collide, dist = self.collisionFree(sig_min)
        if not collide:
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
        S, K = self.LQR.get_sol(x)
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        r = self.gamma*(np.log(len(V))/len(V))**(1/len(x))
        diff = (V - x) # N * d
        argument = diff.T@S@diff
        near = np.linalg.norm(argument, 1) < r
        Xnear = set(map(tuple, V[near]))
        return Xnear

    def LQRNearest(self, V, x):
        S, K = self.LQR.get_sol(x)
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        diff = (V - x) # N * d
        argument = diff.T@S@diff
        xnearest = V[min(argument)]
        return tuple(xnearest)

    def LQRsteer(self, x, x_p):
        # Given two states, x, x'
        #, the LQRSteer(x, x′) function “connects” state x with x′ 
        # using the local LQR policy calculated by linearizing about x.
        pi = self.LQR.lqr_policy(x, x_p) # numpy array of policies
        sigma = self.LinearizedMotionModel(x, pi)
        return sigma

    def cost(self, x):
        # returns the cost in the rrt
        if x == self.x0:
            return 0
        if x not in self.COST:
            if x not in self.Parent:
                self.COST[x] = np.inf
            else:
                self.COST[x] = self.cost(self.Parent[x]) + self.getDist(self.Parent[x], x)
        return self.COST[x]
        


    