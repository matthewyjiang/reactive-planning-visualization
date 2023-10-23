import numpy as np

class task():
    # x: coordinates x in R^d
    # v_star: value gain
    # phi: value function phi in R^d
    undecided_m = 10
    
    def __init__(self, x, v_star, phi, grad):
        self.x = x
        self.v_star = v_star
        self.phi = phi
        self.grad = grad
        
    def get_phi(self, x):
        return self.phi(self.x, x)
    
    def get_grad(self, x):
        return self.grad(self.x, x)