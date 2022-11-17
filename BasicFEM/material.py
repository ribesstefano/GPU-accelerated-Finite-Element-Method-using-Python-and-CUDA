import numpy as np

class Elasticity:
    def __init__(self, E, nu, dim=2, plane_state="plane strain"):
        self.E = E
        self.nu = nu
        if dim == 2 and plane_state == "plane strain":
            stiffness_matrix = E/((1+nu)*(1-2*nu)) * np.array([[1-nu, nu, 0.0], [nu, 1-nu, 0.0], [0.0, 0.0, (1-2*nu)/2]])
        self.stiffness = stiffness_matrix
    
    def material_response(self, epsilon):
        sigma = self.stiffness @ epsilon
        return sigma, self.stiffness