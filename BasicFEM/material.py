import numpy as np
import cupy as cp

class Elasticity:
    def __init__(self, E, nu, dim=2, plane_state='plane strain', device='cpu'):
        self.E = E
        self.nu = nu
        if dim == 2 and plane_state == 'plane strain':
            tmp = np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, (1 - 2 * nu) / 2]])
            stiffness_matrix = E / ((1 + nu) * (1 - 2 * nu)) * tmp
        else:
            raise ValueError(f'Only dim equal to 2 supported. dim={dim} supplied')
        if device == 'gpu':
            self.stiffness = cp.asarray(stiffness_matrix)
        else:
            self.stiffness = stiffness_matrix
    
    def material_response(self, epsilon):
        sigma = self.stiffness @ epsilon
        return sigma, self.stiffness