from BasicFEM import DofHandler, Elasticity, MomentumBalance, Grid, CST

import cProfile
import pstats
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

def meshgrid_triangles(n, m):
    """Returns triangles to mesh a np.meshgrid of n x m points."""
    tri = []
    for i in range(n-1):
        for j in range(m-1):
            a = i + j*(n)
            b = (i+1) + j*n
            d = i + (j+1)*n
            c = (i+1) + (j+1)*n
            if j % 2 == 1:
                tri += [[a, b, d], [b, c, d]]
            else:
                tri += [[a, b, c], [a, c, d]]
    return np.array(tri, dtype=np.int32)

def get_rand_grid(grid_sz):
    """Returns nodes coordinates and cell triangles of a square grid.
    
    Args:
        grid_sz (int): Grid size
    
    Returns:
        Tuple: Nodes and cells as np.ndarray
    """
    xs = np.linspace(-grid_sz / 2, grid_sz / 2, grid_sz)
    nodes = np.vstack([xs, xs]).T
    cells = meshgrid_triangles(grid_sz, grid_sz)
    return nodes, cells

def test_run_simulation(device='cpu'):
    if device == 'gpu':
        xp = cp
    else:
        xp = np
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
    grid = Grid(nodes, cells, device=device)
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
    material = Elasticity(E, nu, device=device)
    # Init weak form equation solver (for single element)
    thickness = 1.0
    weak_form = MomentumBalance(material, thickness)
    # Init and constrain K, f and a components
    I, J = dh.sparsity_pattern(grid)
    if device == 'gpu':
        K = cupyx.scipy.sparse.csr_matrix((xp.zeros(len(I)),
                                          (xp.array(I), xp.array(J))))
    else:
        K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    f = xp.zeros(len(I), dtype=xp.float32)
    a = xp.zeros(dh.ndofs_total(grid), dtype=xp.float32)
    a[top_dofs[:, 1]] = -0.1
    if device == 'gpu':
        bottom_dofs = cp.asarray(bottom_dofs)
        left_dofs = cp.asarray(left_dofs)
        top_dofs = cp.asarray(top_dofs)
    prescribed_dofs = xp.concatenate((bottom_dofs[:, 1],
                                      left_dofs[:,0],
                                      top_dofs[:,1]))
    free_dofs = xp.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)
    # Init local variables for element loop (re-use them across iterations)
    ndofs_cell = dh.ndofs_per_cell(grid)
    dofs = xp.empty(ndofs_cell, dtype=xp.int32)
    ke = xp.zeros((ndofs_cell, ndofs_cell), dtype=xp.float32)
    re = xp.zeros(ndofs_cell, dtype=xp.float32)
    xe = xp.zeros((grid.nnodes_per_cell(), nodes.shape[-1]), dtype=xp.float32)
    element = CST(device=device)
    # Initialize profiler
    profiler = cProfile.Profile()
    # Start profiling
    profiler.enable()
    # Init CuPy profilers
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    # Start CuPy profilers
    start_cpu = time.perf_counter()
    start_gpu.record()
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
    if device == 'gpu':
        a[free_dofs] = cupyx.scipy.sparse.linalg.spsolve(A, b)
    else:
        a[free_dofs] = scipy.sparse.linalg.spsolve(A, b)
    # Stop CuPy profilers
    end_gpu.record()
    end_cpu = time.perf_counter()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu
    print(f'CPU Time: {t_cpu:.3f} ms')
    print(f'GPU Time: {t_gpu:.3f} ms')
    # Stop profiling
    profiler.disable()
    # Create statistics from the profiler, sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # Print the 10 (inclusive) most expensive functions
    stats.print_stats(10)