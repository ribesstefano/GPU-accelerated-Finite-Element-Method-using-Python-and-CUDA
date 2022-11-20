import numpy as np
import cupy as cp

class MomentumBalance:
    def __init__(self, material, thickness):
        """Solving the Weak Form equation for one element. 
        
        Args:
            material (Elasticity): Description
            thickness (float32): Description
        """
        self.material = material
        self.thickness = thickness
    
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

class CST:
    """Constant Strain Triangle.
    
    Description...
    
    Attributes:
        iso_coords (np.ndarray): Adjusted coordinates for...
        weights (np.ndarray): Weight values for...
    """
    def __init__(self, device='cpu'):
        """Inits CST class with default...
        
        Args:
            device (str, optional): Target device for attributes. Options:
            'cpu', 'gpu'. Default: 'cpu'
        """
        iso_coords, weights = self._triangular_quad_points(2, 1, np.float32)
        if device == 'gpu':
            self.iso_coords = cp.asarray(iso_coords)
            self.weights = cp.asarray(weights)
        else:
            self.iso_coords = iso_coords
            self.weights = weights

    def _triangular_quad_points(self, dim, nquadpoints, dtype):
        if dim == 2:
            if nquadpoints == 1:
                iso_coords = np.array([[1/3, 1/3]], dtype=dtype)
                weights = np.array([1/2], dtype=dtype)
                return iso_coords, weights
            elif nquadpoints == 3:
                iso_coords = np.array([[1/6, 1/6],
                                       [2/3, 1/6],
                                       [1/6, 2/3]], dtype=dtype)
                weights = np.array([1/6, 1/6, 1/6], dtype=dtype)
                return iso_coords, weights
        raise ValueError(f'Only dim equal to 2 supported. dim={dim} supplied')

    def jacobian(self, xe):
        xp = cp.get_array_module(xe)
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        x1, x2, x3 = xe
        J = xp.array([[x2[0]-x1[0], x3[0]-x1[0]], [x2[1]-x1[1], x3[1]-x1[1]]])
        return J

    def jacobi_determinant(self, xe):
        xp = cp.get_array_module(xe)
        x1, x2, x3 = xe 
        detJ = (x2[0]-x1[0]) * (x3[1]-x1[1]) - (x3[0]-x1[0]) * (x2[1]-x1[1])
        return detJ

    def shape_values(self, qp):
        xp = cp.get_array_module(self.iso_coords)
        xi1 = iso_coords[qp][0]
        xi2 = iso_coords[qp][1]
        N = xp.array([1.0 - xi1 - xi2, xi1, xi2])
        return N

    def shape_gradients(self, xe):
        xp = cp.get_array_module(xe)
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        Jinv_t = xp.linalg.inv(self.jacobian(xe)).T
        # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
        dNdxi = xp.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        dNdx = xp.empty_like(dNdxi)
        for i in range(dNdx.shape[0]):
            dNdx[i] = Jinv_t @ dNdxi[i]
        return dNdx

    def B_operator(self, xe):
        xp = cp.get_array_module(xe)
        dNdx = self.shape_gradients(xe)
        # TODO(Kim): Not ideal to hard-code dimensions, but will never change
        # for 2D elements...
        B = xp.zeros((3, 2 * dNdx.shape[0]))
        for i in range(dNdx.shape[0]):
            dNidx = dNdx[i][0]
            dNidy = dNdx[i][1]
            B[0, 2 * i] = dNidx
            B[2, 2 * i] = dNidy
            B[1, 2 * i + 1] = dNidy
            B[2, 2 * i + 1] = dNidx
        return B