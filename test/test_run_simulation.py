from BasicFEM import DofHandler, Elasticity, MomentumBalance, Grid, CST

import cProfile
import pstats
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def test_run_simulation():
    # Define grid and cells
    nodes = np.array([[-1., -1.],
                      [ 1., -1.],
                      [ 1.,  1.],
                      [-1.,  1.],
                      [ 0.,  0.]], dtype=np.float32)
    cells = [(0,1,4), (1,2,4), (4,2,3), (0,4,3)]
    cells = np.array(cells, dtype=np.int32)
    # TODO(Stefano): Work in progress on automatic random grid generation
    # nodes, cells = get_rand_grid(grid_sz=8)
    grid = Grid(nodes, cells)
    # TODO(Kim): The following should be stored in grid
    bottom_nodes = [0, 1]
    left_nodes = [0, 3]
    top_nodes = [2, 3]
    # Init Degree of Freedom handler
    dh = DofHandler(2, grid)
    # Constrained nodes
    bottom_dofs = dh.getdofs(bottom_nodes).astype(np.int32)
    left_dofs = dh.getdofs(left_nodes).astype(np.int32)
    top_dofs = dh.getdofs(top_nodes).astype(np.int32)
    # Init material specifications
    E = 200e3
    nu = 0.3
    material = Elasticity(E, nu)
    # Init weak form equation solver (for single element)
    thickness = 1.0
    weak_form = MomentumBalance(material, thickness)
    # Init and constrain K, f and a components
    I, J = dh.sparsity_pattern(grid)
    K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    f = np.zeros(len(I), dtype=np.float32)
    a = np.zeros(dh.ndofs_total(grid), dtype=np.float32)
    a[top_dofs[:, 1]] = -0.1
    prescribed_dofs = np.concatenate((bottom_dofs[:, 1],
                                      left_dofs[:,0],
                                      top_dofs[:,1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)
    # Init local variables for element loop (re-use them across iterations)
    ndofs_cell = dh.ndofs_per_cell(grid)
    dofs = np.empty(ndofs_cell, dtype=np.int32)
    ke = np.zeros((ndofs_cell, ndofs_cell), dtype=np.float32)
    re = np.zeros(ndofs_cell, dtype=np.float32)
    xe = np.zeros((grid.nnodes_per_cell(), nodes.shape[-1]), dtype=np.float32)
    element = CST()
    # Initialize profiler
    profiler = cProfile.Profile()
    # Start profiling
    profiler.enable()
    # Assemble K and f components
    for cellid in range(len(grid.cells)):
        # NOTE(Stefano): The methods getcoordinates and celldofs now return the
        # updated values. Makes the code more readable and Pythonic. Also, it
        # doesn't affect performance (it's just passing around references, i.e.
        # pointers, not the actual values).
        xe = grid.getcoordinates(xe, cellid)
        dofs = dh.celldofs(dofs, grid, cellid)
        ke.fill(0.0)
        re.fill(0.0)
        ue = a[dofs] # Not relevant as input for linear elasticity
        weak_form.element_routine(ke, re, element, xe, ue)
        # TODO(Kim): Probably there is a better way for doing this
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += ke[i, j]
            f[dof_i] += re[i]
    A = K[free_dofs, :][:, free_dofs]
    b = -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs]
    a[free_dofs] = scipy.sparse.linalg.spsolve(A, b)
    # Stop profiling
    profiler.disable()
    # Create statistics from the profiler, sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # Print the 10 (inclusive) most expensive functions
    stats.print_stats(10)