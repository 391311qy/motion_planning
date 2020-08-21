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
        self.gamma = 10e+5
        self.maxiter = n

        self.S_mat = {}

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

    def MotonModel(self, x, u):
        '''real motion model'''
        return self.quad.MotionModel(x, u)

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
        A_g, B_g = self.LQR.Linearized_Motion_Model(goal, np.zeros(4))
        Q, R = self.LQR.Continuous_Cost_Model()
        S_g, K_g = self.LQR.lqr_solve(A_g,B_g,Q,R)
        xn = self.LQRNear(self.V, goal, S_g)
        c = [self.cost(tuple(x)) for x in xn]
        xn = np.array(xn)
        xncmin = xn[np.argmin(c)]
        self.wireup(goal, tuple(xncmin))
        self.Path = self.path()
        print(self.Path)

##################################### LQR RRT star implementation    
    def LQR_rrt_star(self, bias = 0.05):
        self.V.add(self.env.start)
        A_0, B_0 = self.LQR.Linearized_Motion_Model(self.env.start, np.zeros(4))
        Q, R = self.LQR.Continuous_Cost_Model()
        S_0, K_0 = self.LQR.lqr_solve(A_0, B_0, Q, R)
        self.S_mat[self.env.start] = S_0
        while self.ind < self.maxiter:
            visualization(self)
            print(self.ind)
            xrand = self.env.sampleFree(bias)
            # lqr linearized at xrand
            A_rand, B_rand = self.LQR.Linearized_Motion_Model(xrand, np.array([0, 0, 0, 0]))
            S_rand, K_rand = self.LQR.lqr_solve(A_rand, B_rand, Q, R)
            # find the nearest point based on cost
            xnearest = self.LQRNearest(self.V, xrand, S_rand)
            pi = -K_rand@(np.array(xnearest) - np.array(xrand)).T     
            xnew, pi = self.LQRsteer(xnearest, xrand, K_rand)
            collide = self.collisionFree(xnearest, xnew)
            if not collide:
                self.V.add(xnew)
                A_new, B_new = self.LQR.Linearized_Motion_Model(xnew, np.array([0, 0, 0, 0]))
                S_new, K_new = self.LQR.lqr_solve(A_new, B_new, Q, R)
                self.S_mat[xnew] = S_new
                xmin, cmin = xnearest, self.cost(xnearest) + self.LQRcost(xnearest, xnew, S_new)
                Xnear = self.LQRNear(self.V, xnew, S_new)
                print(Xnear)
                # xmin, cmin = xnearest, self.cost(xnearest) + self.env.getDist(xnearest, xnew)
                Collide = []
                # for xnear in Xnear:
                #     xnear = tuple(xnear)
                #     c1 = self.cost(xnear) + self.LQRcost(xnew, xnear, self.S_mat[xnear])
                #     # c1 = self.cost(xnear) + self.env.getDist(xnew, xnear)
                #     collide = self.collisionFree(xnew, xnear)
                #     Collide.append(collide)
                #     if not collide and c1 < cmin:
                #         xmin , cmin = xnear, c1
                self.wireup(xnew, xmin)
                # for i in range(len(Xnear)):
                #     collide = Collide[i]
                #     xnear = tuple(Xnear[i])
                #     c2 = self.cost(xnew) + self.LQRcost(xnew, xnear, self.S_mat[xnear])
                #     # c2 = self.cost(xnew) + self.env.getDist(xnew, xnear)
                #     if not collide and c2 < self.cost(xnear):
                #         self.removewire(self.Parent[xnear], xnear)
                #         self.wireup(xnear, xnew)
            self.ind += 1
        self.reached()
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

    def LQRNear(self, V, x, S_x, r=None):
        V = np.array(list(V))
        if len(V) == 1:
            return V
        # S, K = self.LQR.get_sol(x)
        x = repmat(np.array(x), len(V), 1)
        if r is None:
            r = self.gamma*(np.log(len(V))/len(V))**(1/len(x))
        if self.done:
            r = 10000
        diff = (V - x) # N * d
        argument = [i@S_x@i.T for i in diff]
        near = np.linalg.norm(argument, 1) < r
        Xnear = np.squeeze(V[near])
        return np.array(Xnear)

    def LQRNearest(self, V, x, S_x):
        if len(V) == 1:
            return list(V)[0]
        # S, K = self.LQR.get_sol(x)
        V = np.array(list(V))
        x = repmat(np.array(x), len(V), 1)
        diff = (V - x) # N * d
        argument = [i@S_x@i.T for i in diff]
        xnearest = V[np.argmin(argument)]
        return tuple(xnearest)

    def LQRsteer(self, x, x_p, K_x_p):
        '''connects two states together. Using the dynamics model of the state space, (not the linearized version)
        but the control signal from the LQR'''
        
        if K_x_p is None:
            A, B = self.LQR.Linearized_Motion_Model(x_p, np.array([0, 0, 0, 0]))
            Q, R = self.LQR.Continuous_Cost_Model()
            S, K_x_p = self.LQR.lqr_solve(A, B, Q, R)
        diff = np.subtract(x_p, x)
        diff = self.restrict_diff(diff, q1=x[3:7], q2=x_p[3:7], t = 0.3)
        x_p = np.add(x, diff)
        A, B = self.LQR.Linearized_Motion_Model(x_p, np.array([0, 0, 0, 0]))
        Q, R = self.LQR.Continuous_Cost_Model()
        S, K_x_p = self.LQR.lqr_solve(A, B, Q, R)
        pi = -K_x_p@diff.T 
        pi = self.restrict_pi(pi)
        # xdot = self.MotonModel(x, pi)
        xdot = self.LinearizedMotionModel(diff, pi)
        xnew = np.add(xdot, x) 
        return tuple(xnew), pi

    def LQRcost(self, x1, x2, Sx2):
        # cost from x1 to x2
        diff = np.array(x2) - np.array(x1)
        return diff@Sx2@diff.T

    def cost(self, x):
        # recursive cost to come
        if x == self.env.start:
            return 0
        lqrcost = self.LQRcost(self.Parent[x], x, self.S_mat[x])
        return self.cost(self.Parent[x]) + lqrcost


    def restrict_pi(self, pi):
        # restricting inputs according to the control limits
        return self.quad.control_restriction(pi)

    def isvalid(self, pi):
        '''check if an action is valid or not'''
        for i in range(0,3):
            if not self.quad.w_range[0]<= pi[i] < self.quad.w_range[1]:
                return False
        if not self.quad.c_range[0]<= pi[3] < self.quad.c_range[1]:
            return False
        return True
    
    def restrict_diff(self, diff, q1, q2, t = 0.1):
        # restricting inputs for the difference between the states for linearized model
        # important: manully set the step size.
        return self.quad.state_restriction(diff, q1, q2, slerp_t=t)

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
    session = lqrrt(500)
    V, E = session.LQR_rrt_star(bias = 0.05)
    # quad = Quadrotor()
    # pose_set = [quad.state_to_OBB(v) for v in V]

        


    