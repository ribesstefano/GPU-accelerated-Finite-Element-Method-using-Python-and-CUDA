import numpy as np
from BasicFEM import *
import scipy.sparse
import scipy.sparse.linalg

def test_run_simulation():
    nodes = np.array([  [-1., -1.],
                        [ 1., -1.],
                        [ 1.,  1.],
                        [-1.,  1.],
                        [ 0.,  0.]])
    cells = [(0,1,4), (1,2,4), (4,2,3), (0,4,3)]
    grid = Grid(nodes, cells, None)
    # should be stored in grid
    bottom_nodes = [0, 1]
    left_nodes = [0, 3]
    top_nodes = [2, 3]

    dh = DofHandler(2)

    E = 200e3
    nu = 0.3
    material = Elasticity(E, nu)

    thickness = 1.0
    weak_form = MomentumBalance(material, thickness)

    element = CST()

    I, J = dh.sparsity_pattern(grid)
    K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    f = np.zeros(len(I))

    ndofs_cell = dh.ndofs_per_cell(grid)
    ke = np.zeros((ndofs_cell, ndofs_cell))
    re = np.zeros(ndofs_cell)
    dofs = np.empty(ndofs_cell, np.int)
    xe = np.zeros((grid.nnodes_per_cell(), 2))
    a = np.zeros(dh.ndofs_total(grid))

    bottom_dofs = dh.getdofs(bottom_nodes)
    left_dofs = dh.getdofs(left_nodes)
    top_dofs = dh.getdofs(top_nodes)

    a[top_dofs[:,1]] = -0.1

    for cellid in range(len(grid.cells)):
        grid.getcoordinates(xe, cellid)
        dh.celldofs(dofs, grid, cellid)
        ke.fill(0.0)
        re.fill(0.0)
        ue = a[dofs] # not relevant as input for linear elasticity
        weak_form.element_routine(ke, re, element, xe, ue)
        # probably there is a better way for this
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += ke[i,j]
            f[dof_i] += re[i]

    prescribed_dofs = np.concatenate((bottom_dofs[:, 1], left_dofs[:,0], top_dofs[:,1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)

    a[free_dofs] = scipy.sparse.linalg.spsolve(K[free_dofs, :][:, free_dofs], -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs])

    




def test_run_simulation_plate_with_hole():
    grid = generate_grid(0.1)

    dh = DofHandler(2)

    E = 200e3
    nu = 0.3
    material = Elasticity(E, nu)

    thickness = 1.0
    weak_form = MomentumBalance(material, thickness)

    element = CST()

    I, J = dh.sparsity_pattern(grid)
    K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    f = np.zeros(len(I))

    ndofs_cell = dh.ndofs_per_cell(grid)
    ke = np.zeros((ndofs_cell, ndofs_cell))
    re = np.zeros(ndofs_cell)
    dofs = np.empty(ndofs_cell, np.int)
    xe = np.zeros((grid.nnodes_per_cell(), 2))
    a = np.zeros(dh.ndofs_total(grid))

    bottom_dofs = dh.getdofs(grid.nodesets['bottom'])
    left_dofs = dh.getdofs(grid.nodesets['left'])
    top_dofs = dh.getdofs(grid.nodesets['top'])

    a[top_dofs[:,1]] = -0.1

    for cellid in range(len(grid.cells)):
        grid.getcoordinates(xe, cellid)
        dh.celldofs(dofs, grid, cellid)
        ke.fill(0.0)
        re.fill(0.0)
        ue = a[dofs] # not relevant as input for linear elasticity
        weak_form.element_routine(ke, re, element, xe, ue)
        # probably there is a better way for this
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += ke[i,j]
            f[dof_i] += re[i]

    prescribed_dofs = np.concatenate((bottom_dofs[:, 1], left_dofs[:,0], top_dofs[:,1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)

    a[free_dofs] = scipy.sparse.linalg.spsolve(K[free_dofs, :][:, free_dofs], -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs])

