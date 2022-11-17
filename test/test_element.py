import pytest
import numpy as np
from BasicFEM import *

def test_material():
    material = Elasticity(200e3, 0.3)
    sigma, dsde = material.material_response(np.zeros(3))
    assert np.array_equal(sigma, np.zeros(3))

def test_shape_values():
    element = CST()
    N = element.shape_values(0)
    assert np.isclose(N, np.ones(3)/3).all()

def test_element():
    # material parameters for stress-strain relation
    K = 1.25
    G = 0.4
    E = 9*K*G/(3*K + G)
    nu = (3*K-2*G)/(6*K+2*G)

    material = Elasticity(E, nu)
    thickness = 2.0
    # weak form that should be assembled
    weak_form = MomentumBalance(material, thickness)
    # interpolations + quadrature rule --> element
    element = CST()
    # element-wise FE-matrices
    ke = np.zeros((6,6))
    re = np.zeros(6)
    # coordinates for this specific element
    xe = np.array([[6., -4.], [5., 3.], [-4., -1.]])
    # zero displacements
    ue = np.zeros(6)
    # assemble matrices
    weak_form.element_routine(ke, re, element, xe, ue)
    # sample solution to compare to
    ke_solution = np.array([  [0.9095, -0.7433, -0.2179, 0.4259, -0.6915, 0.3174],
                [-0.7433, 2.2515, -0.1575, -2.3239, 0.9007, 0.0724],
                [-0.2179, -0.1575, 0.8366, 0.6194, -0.6187, -0.4619],
                [0.4259, -2.3239, 0.6194, 2.7154, -1.0453, -0.3915],
                [-0.6915, 0.9007, -0.6187, -1.0453, 1.3102, 0.1445,],
                [0.3174, 0.0724, -0.4619, -0.3915, 0.1445, 0.3192]])
    assert np.array_equal(ke_solution, ke.round(4))
    assert np.array_equal(np.zeros(6), re)

    # non-zero displacements
    ke.fill(0.0)
    re.fill(0.0)
    ue = np.random.rand(6)
    weak_form.element_routine(ke, re, element, xe, ue)
    assert np.array_equal(ke_solution, ke.round(4))
    assert np.isclose(re, ke @ ue).all()

