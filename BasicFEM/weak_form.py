import numpy as np

class MomentumBalance:
    def __init__(self, material, thickness, element=None):
        """Solving the Weak Form equation for one element. 
        
        Args:
            material (Elasticity): Description
            thickness (float32): Description
        """
        self.material = material
        self.thickness = thickness
        if element:
            self.element = element
        else:
            self.element = CST()
    
    def element_routine(self, ke, re, element, xe, ue):
        B = element.B_operator(xe)
        detJ = element.jacobi_determinant(xe)
        nqp = element.weights.shape[0]
        for qp in range(0, nqp):
            w = element.weights[qp]
            epsilon = B @ ue
            sigma, dsde = self.material.material_response(epsilon)
            re += B.T @ sigma * detJ * w
            ke += B.T @ dsde @ B * detJ * w
        re *= self.thickness
        ke *= self.thickness
        # TODO(Kim): Missing Neumann boundary condition

    def run_element_routine(self, ke, re, xe, ue):
        B = self.element.B_operator(xe)
        detJ = self.element.jacobi_determinant(xe)
        nqp = self.element.weights.shape[0]
        for qp in range(0, nqp):
            w = self.element.weights[qp]
            epsilon = B @ ue
            sigma, dsde = self.material.material_response(epsilon)
            re += B.T @ sigma * detJ * w
            ke += B.T @ dsde @ B * detJ * w
        re *= self.thickness
        ke *= self.thickness
        # TODO(Kim): Missing Neumann boundary condition

class CST:
    """Constant Strain Triangle.
    
    Description...
    
    Attributes:
        iso_coords (np.ndarray): Adjusted coordinates for...
        weights (np.ndarray): Weight values for...
    """
    def __init__(self, dtype=np.float32):
        """Inits CST class with default...
        
        Args:
            device (str, optional): Target device for attributes. Options:
            'cpu', 'gpu'. Default: 'cpu'
        """
        self.dtype = dtype
        iso_coords, weights = self._triangular_quad_points(dim=2, n_quadpoints=1)
        self.iso_coords = iso_coords
        self.weights = weights

    def _triangular_quad_points(self, dim, n_quadpoints):
        if dim == 2:
            if n_quadpoints == 1:
                iso_coords = np.array([[1 / 3, 1 / 3]], dtype=self.dtype)
                weights = np.array([1 / 2], dtype=self.dtype)
                return iso_coords, weights
            elif n_quadpoints == 3:
                iso_coords = np.array([[1 / 6, 1 / 6],
                                       [2 / 3, 1 / 6],
                                       [1 / 6, 2 / 3]], dtype=self.dtype)
                weights = np.array([1 / 6, 1 / 6, 1 / 6], dtype=self.dtype)
                return iso_coords, weights
        raise ValueError(f'Only dim equal to 2 supported. dim={dim} supplied')

    def jacobian(self, xe):
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        x1, x2, x3 = xe
        J = np.array([[x2[0]-x1[0], x3[0]-x1[0]], [x2[1]-x1[1], x3[1]-x1[1]]])
        return J

    def jacobi_determinant(self, xe):
        x1, x2, x3 = xe 
        detJ = (x2[0]-x1[0]) * (x3[1]-x1[1]) - (x3[0]-x1[0]) * (x2[1]-x1[1])
        return detJ

    def shape_values(self, qp):
        xi1 = self.iso_coords[qp][0]
        xi2 = self.iso_coords[qp][1]
        N = np.array([1.0 - xi1 - xi2, xi1, xi2])
        return N

    def shape_gradients(self, xe):
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        Jinv_t = np.linalg.inv(self.jacobian(xe)).T
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        dNdxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=self.dtype)
        dNdx = np.empty_like(dNdxi)
        for i in range(dNdx.shape[0]):
            dNdx[i] = Jinv_t @ dNdxi[i]
        return dNdx

    def B_operator(self, xe):
        dNdx = self.shape_gradients(xe)
        # TODO(Kim): Not ideal to hard-code dimensions, but will never change
        # for 2D elements...
        B = np.zeros((3, 2 * dNdx.shape[0]), dtype=self.dtype)
        for i in range(dNdx.shape[0]):
            dNidx = dNdx[i][0]
            dNidy = dNdx[i][1]
            B[0, 2 * i] = dNidx
            B[2, 2 * i] = dNidy
            B[1, 2 * i + 1] = dNidy
            B[2, 2 * i + 1] = dNidx
        return B
