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
        A += 10e-6
        B += 10e-6
        S = la.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R)@(B.T@S)
        return S, K

    def lqr_policy(self, x, x_p, K_x_p):
        # get the policy from x to x_p
        x = np.array(x)
        x_p = np.array(x_p)
        x_bar = x - x_p
        policy = -(K_x_p@x_bar)
        return policy

if __name__ == "__main__":
    pass

   

    
