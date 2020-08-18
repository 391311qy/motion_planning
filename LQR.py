import numpy as np
import numpy.linalg as npl
import scipy.linalg as la

# solve the riccatti equation given following matrix inputs
# A: df(x, u)/dx
# B: df(x, u)/du
# Q: Q
# R: R in J = sum: 0-> inf (xTQx + uTRu)
from quadrotor import Quadrotor

class lqr:

    def __init__(self):
        # records LQR solution at every point
        '''
        params: 
        len_x: input state length
        len_u: input control length
        S_set: sln to riccatti eqn
        K_set: LQR gain
        Q: quadratic cost for state
        R: quadratic cost for control
        '''
        self.len_x = 10
        self.len_u = 4
        self.S_set = {}
        self.K_set = {}
        self.Q, self.R = self.Continuous_Cost_Model()
        self.quad = Quadrotor()

    def Linearized_Motion_Model(self, x0, u0):
        # return A(x0, u0), B(x0, u0)
        '''
        params:
        x0, u0: reference state and reference control
        returns: 
        A: linearized transition matrix
        B: linearized control matrix
        x_ = A(.)(x-x0) + B(.)(u-u0) is linearized motion model 
        '''
        A = self.quad.linearized_A(x0, u0)
        B = self.quad.linearized_B(x0, u0)
        return A, B

    def Continuous_Cost_Model(self):
        # return Q, R
        Q = np.diag([100, 100, 100, 1, 1, 1, 1, 10, 10, 10])
        R = np.diag([1, 5, 5, 0.1])
        return Q, R

    def lqr_solve(self, A, B, Q, R):
        # S, K = self.riccattiSolve(A, B, Q, R)
        A += 10e-12
        B += 10e-12
        S = la.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R)@(B.T@S)
        return S, K

    def get_sol(self, x):
        # get the stored solution of CARE
        if x not in self.S_set:
            Q, R = self.Q, self.R
            A, B = self.Linearized_Motion_Model(x, self.quad.control_restriction(np.zeros(self.len_u)))
            S, K = self.lqr_solve(A, B, Q, R)
            self.S_set[x] = S
            self.K_set[x] = K
            return S, K
        return self.S_set[x], self.K_set[x]

    def lqr_policy(self, x, x_p):
        S, K = self.get_sol(x)
        x = np.array(x)
        x_p = np.array(x_p)
        x_bar = x_p - x
        policy = -(K@x_bar)
        return policy

    def care(self, A, B, Q, R):
        # continuous algebriac riccati equation solver
        # TODO: need to solve CARE.
        pass


    
