'''
@ author: yue qi
lqRRTstar for optimal kinodynamic motion planning, possibly using ROS in future

source: 
Webb, Dustin J., and Jur Van Den Berg. 
"Kinodynamic RRT*: Asymptotically optimal motion planning for robots with linear dynamics." 
2013 IEEE International Conference on Robotics and Automation. IEEE, 2013.
'''

import numpy as np
import numpy.linalg as npl

from env3D import env

class lqrrt:
    def __init__(self):
        self.env = env()