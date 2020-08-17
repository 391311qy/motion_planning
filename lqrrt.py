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

# from inv_pendulum import env
from quadrotor import env, Quadrotor
from LQR import lqr


class lqrrt:

    def __init__(self):
        self.env = env()
        self.V = set() # set of positions
        self.E = set() # set of edges
        self.X = set() # set of states
        self.Parent = {}
        self.gamma = 10000
        self.LQR = lqr()
        self.COST = {}
        self.STAGECOST = {}
        self.maxiter = 100

    def LinearizedMotionModel(self, x, u):
        # x: tuple state (x-x0)
        # u: array contorl policy (u-u0)
        A, B = self.LQR.Linearized_Motion_Model(x, u)
        x = np.array(x)
        u = np.array(u)
        return A@x + B@u
        # define A(x0,u0), B(x0,u0), Q, R here

    def collisionFree(self, x1):
        # x2 = (0,0,0) # center of the inverted pendulum
        # pos1, pos2 = x1[0:3], x2[0:3] # still tuple
        # collide = self.env.isCollide(pos1, pos2, dist = self.env.inv_pen.r)
        collide = self.env.isCollide(x1)
        return collide

##################################### LQR RRT star implementation    
    def LQR_rrt_star(self):
        self.V.add(self.env.start)
        for i in range(self.maxiter):
            print(i)
            xrand = self.env.sampleFree()
            xnearest = self.LQRNearest(self.V, xrand)
            xnew, pi = self.LQRsteer(xnearest, xrand)
            # Xnear = self.LQRNear(self.V, xnew)
            # xmin, sig_min = self.ChooseParent(Xnear, xnew)
            # collide = self.collisionFree(sig_min)
            collide = self.collisionFree(xnew)
            if not collide:
                self.V.add(xnew)
                self.E.add((xnearest, xnew))
                # self.E.add((xmin, xnew))
                # self.Parent[xnew] = xmin
                # self.V, self.E = self.rewire(self.V, self.E, Xnear, xnew)
        return self.V, self.E

    def ChooseParent(self, Xnear, xrand):
        minCost = np.inf
        xmin = None
        sig_min = None
        for xnear in Xnear:
            sigma, pi = self.LQRsteer(xnear, xrand)
            newcost = self.cost(xnear) + self.cost(sigma)
            print(self.cost(sigma))
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
        if len(V) == 1:
            return V
        S, K = self.LQR.get_sol(x)
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        # r = self.gamma*(np.log(len(V))/len(V))**(1/len(x))
        r = 10000000
        diff = (V - x) # N * d
        argument = [i@S@i.T for i in diff]
        near = np.linalg.norm(argument, 1) < r
        Xnear = set(map(tuple, V[near]))
        return Xnear

    def LQRNearest(self, V, x):
        if len(V) == 1:
            return list(V)[0]
        S, K = self.LQR.get_sol(x)
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        diff = (V - x) # N * d
        argument = [i@S@i.T for i in diff]
        xnearest = V[np.argmin(argument)]
        return tuple(xnearest)

    def LQRsteer(self, x, x_p):
        # Given two states, x, x'
        #, the LQRSteer(x, x′) function “connects” state x with x′ 
        # using the local LQR policy calculated by linearizing about x.
        pi = self.LQR.lqr_policy(x, x_p) # numpy array of policies
        sigma = self.LinearizedMotionModel(np.subtract(x_p, x), pi) # increment on movement
        return tuple(sigma + np.array(x)), pi

    def cost(self, x):
        # returns the cost in the rrt
        if x == self.env.start:
            return 0
        if x not in self.COST:
            S, K = self.LQR.get_sol(x) # TODO: verify if it is at xparent or x.
            diff = np.array(self.Parent[x]) - np.array(x)
            LQRcost = diff@S@diff.T # (v-x)TS(v-s)
            self.COST[x] = self.cost(self.Parent[x]) + LQRcost
        return self.COST[x]

if __name__ == '__main__':
    session = lqrrt()
    V, E = session.LQR_rrt_star()
    quad = Quadrotor()
    pose_set = [quad.state_to_OBB(v) for v in V]

        


    