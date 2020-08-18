'''
@ author: yue qi
lqRRTstar for optimal kinodynamic motion planning, possibly using ROS in future

source:
Perez, Alejandro, et al. 
"LQR-RRT*: Optimal sampling-based motion planning with automatically derived extension heuristics." 
2012 IEEE International Conference on Robotics and Automation. IEEE, 2012.
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
from numpy.matlib import repmat

# from inv_pendulum import env
from quadrotor import env, Quadrotor, visualization, q_L
from LQR import lqr


class lqrrt:

    def __init__(self, n = 1000):
        self.env = env()
        self.quad = Quadrotor()
        self.LQR = lqr()
        self.V = set() # set of vertices
        self.E = set() # set of edges
        self.Parent = {}
        self.COST = {}
        self.STAGECOST = {}
        self.gamma = 10000
        self.maxiter = n


        self.done = False
        self.ind = 0
        self.Path = []

    def LinearizedMotionModel(self, x, u):
        # x: tuple state (x-x0)
        # u: array contorl policy (u-u0)
        A, B = self.LQR.Linearized_Motion_Model(x, u)
        x = np.array(x)
        u = np.array(u)
        return A@x + B@u
        # define A(x0,u0), B(x0,u0), Q, R here

    def collisionFree(self, x1, x2):
        # x2 = (0,0,0) # center of the inverted pendulum
        # pos1, pos2 = x1[0:3], x2[0:3] # still tuple
        # collide = self.env.isCollide(pos1, pos2, dist = self.env.inv_pen.r)
        collide = self.env.isCollide(x1, x2)
        return collide

    def wireup(self, x, y):
        self.E.add((x,y)) # add edge
        self.Parent[x] = y

    def removewire(self, x, y):
        if (x, y) in self.E:
            self.E.remove((x, y))

    def reached(self):
        self.done = True
        goal = self.env.goal
        xn = self.LQRNear(self.V, goal)
        c = [self.cost(x) for x in xn]
        xn = np.array(xn)
        xncmin = xn[np.argmin(c)]
        self.wireup(goal, xncmin)
        self.Path = self.path()

##################################### LQR RRT star implementation    
    def LQR_rrt_star(self):
        self.V.add(self.env.start)
        while self.ind < self.maxiter:
            print(self.ind)
            xrand = self.env.sampleFree()
            xnearest = self.LQRNearest(self.V, xrand)
            xnew, pi = self.LQRsteer(xnearest, xrand)
            collide = self.collisionFree(xnearest, xnew)
            if not collide:
                Xnear = self.LQRNear(self.V, xnew)
                self.V.add(xnew)
                xmin, cmin = xnearest, self.cost(xnearest) + self.LQRcost(xnearest, xnew)
                # xmin, cmin = xnearest, self.cost(xnearest) + self.env.getDist(xnearest, xnew)
                Collide = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    # c1 = self.cost(xnear) + self.LQRcost(xnew, xnear)
                    c1 = self.cost(xnear) + self.env.getDist(xnew, xnear)
                    collide = self.collisionFree(xnew, xnear)
                    Collide.append(collide)
                    if not collide and c1 < cmin:
                        xmin , cmin = xnear, c1
                self.wireup(xnew, xmin)
                for i in range(len(Xnear)):
                    collide = Collide[i]
                    xnear = tuple(Xnear[i])
                    c2 = self.cost(xnew) + self.LQRcost(xnew, xnear)
                    # c2 = self.cost(xnew) + self.env.getDist(xnew, xnear)
                    if not collide and c2 < self.cost(xnear):
                        self.removewire(self.Parent[xnear], xnear)
                        self.wireup(xnear, xnew)
            self.ind += 1
        # self.reached()
        visualization(self)
        plt.show()
        return self.V, self.E

    # def ChooseParent(self, Xnear, xrand):
    #     minCost = np.inf
    #     xmin = None
    #     sig_min = None
    #     for xnear in Xnear:
    #         sigma, pi = self.LQRsteer(xnear, xrand)
    #         newcost = self.cost(xnear) + self.cost(sigma)
    #         if newcost < minCost:
    #             minCost, xmin = newcost, xnear
    #             sig_min = sigma
    #     return xmin, sig_min

    # def rewire(self, V, E, Xnear, xnew):
    #     for xnear in Xnear:
    #         sig = self.LQRsteer(xnew, xnear)
    #         if self.cost(xnew) + self.cost(sig) < self.cost(xnear):
    #             if self.collisionFree(xnew, sig):
    #                 xparent = self.Parent[xnear]
    #                 E.remove((xparent, xnear))
    #                 E.add((xnew, xnear))
    #     return V, E

    def LQRNear(self, V, x):
        V = np.array(list(V))
        
        S, K = self.LQR.get_sol(x)
        if len(V) == 1:
            return V
        x = repmat(np.array(x), len(V), 1)
        # r = self.gamma*(np.log(len(V))/len(V))**(1/len(x))
        r = 10000000
        diff = (V - x) # N * d
        argument = [i@S@i.T for i in diff]
        near = np.linalg.norm(argument, 1) < r
        Xnear = np.squeeze(V[near])
        return np.array(Xnear)

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
        viable_pi = self.restrict(pi)
        diff = np.subtract(x_p, x)
        sigma = self.LinearizedMotionModel(diff, viable_pi) # increment on movement
        newpose = sigma + np.array(x)
        # return tuple(sigma + np.array(x)), pi
        return tuple(newpose), pi

    def LQRcost(self, x1, x2):
        # cost from x1 to x2
        S, K = self.LQR.get_sol(x1) # TODO: verify if it is at xparent or x.
        diff = np.array(x2) - np.array(x1)
        return diff@S@diff.T

    def cost(self, x):
        # recursive cost to come
        if x == self.env.start:
            return 0
        lqrcost = self.LQRcost(self.Parent[x], x)
        return self.cost(self.Parent[x]) + lqrcost
        # return self.cost(self.Parent[x]) + self.env.getDist(self.Parent[x], x)

    def restrict(self, pi):
        # restricting inputs according to the control limits
        return self.quad.control_restriction(pi)

    def path(self):
        path = []
        x = self.env.goal
        i = 0
        while x != self.env.start:
            path.append((self.Parent[x],x))
            x = self.Parent[x]
            i+=1
            if i > 10000:
                break 
        return path


if __name__ == '__main__':
    session = lqrrt(1000)
    V, E = session.LQR_rrt_star()
    # quad = Quadrotor()
    # pose_set = [quad.state_to_OBB(v) for v in V]

        


    