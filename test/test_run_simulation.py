from BasicFEM import * 

import cProfile
import pstats
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import timeit

import pymetis


def create_adjacency_matrix(connections):
    matrix = defaultdict(dict)
    for a, b in connections:
        matrix[a][b] = 1
        matrix[b][a] = 1
    return matrix


def is_connected_to_all(vertex, group, matrix):
    for v in group:
        if vertex != v and vertex not in matrix[v]:
            return False
    return True


def group_vertexes(vertixes):
    matrix = create_adjacency_matrix(vertixes)
    groups = []
    current_group = set()
    for vertex in matrix.keys():
        if is_connected_to_all(vertex, current_group, matrix):
            current_group.add(vertex)
        else:
            groups.append(current_group)
            current_group = {vertex}
    groups.append(current_group)
    return groups


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
    f_len = len(I)
    a_len = dh.ndofs_total(grid)
    f = np.zeros(f_len, dtype=np.float32)
    a = np.zeros(a_len, dtype=np.float32)
    a[top_dofs[:, 1]] = -0.1
    # K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    K = np.zeros((a_len, f_len), dtype=np.float32)
    prescribed_dofs = np.concatenate((bottom_dofs[:, 1],
                                      left_dofs[:, 0],
                                      top_dofs[:, 1]))
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
        xe = grid.getcoordinates(cellid, xe)
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

    prescribed_dofs = np.concatenate((bottom_dofs[:, 1], left_dofs[:, 0], top_dofs[:, 1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)

    A = scipy.sparse.csr_matrix(K[free_dofs, :][:, free_dofs], dtype=np.float32, copy=True)
    Kb = scipy.sparse.csr_matrix(-K[free_dofs, :][:, prescribed_dofs], dtype=np.float32, copy=True)
    b = Kb @ a[prescribed_dofs]
    a[free_dofs] = scipy.sparse.linalg.spsolve(A, b)
    # a[free_dofs] = scipy.sparse.linalg.spsolve(K[free_dofs, :][:, free_dofs], -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs])

    # Stop profiling
    profiler.disable()
    # Create statistics from the profiler, sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # Print the 10 (inclusive) most expensive functions
    stats.print_stats(100)
    stats.dump_stats('program.prof')


import matplotlib.pyplot as plt
import matplotlib.tri as tri

def test_run_simulation_plate_with_hole():
    # ==========================================================================
    # Define mesh
    # ==========================================================================
    # grid = generate_grid(0.1) # Original
    grid = generate_grid(lcar=0.8)
    dh = DofHandler(n_dofs_per_node=2, grid=grid)
    # ==========================================================================
    # Define a and prescribe DoFs
    # ==========================================================================
    a = np.zeros(dh.get_ndofs_total())
    bottom_dofs = dh.get_nodes_dofs(grid.nodesets['bottom'])
    left_dofs = dh.get_nodes_dofs(grid.nodesets['left'])
    top_dofs = dh.get_nodes_dofs(grid.nodesets['top'])
    prescribed_dofs = (bottom_dofs[:, 1], left_dofs[:, 0], top_dofs[:, 1])
    prescribed_dofs = np.concatenate(prescribed_dofs)
    free_dofs = np.setdiff1d(range(dh.get_ndofs_total()), prescribed_dofs)
    a[top_dofs[:, 1]] = -0.1
    # ==========================================================================
    # Plot mesh
    # ==========================================================================
    # grid.plot()
    # grid.plot(np.random.randn(len(grid.nodes))) # With colored rand nodal data
    # ==========================================================================
    # Naive global K assembly
    # ==========================================================================
    # Init global K matrix and f
    row_idx, col_idx = dh.get_sparsity_pattern()
    
    # # subgraphs = group_vertexes([(a, b) for a, b in zip(row_idx, col_idx)])
    # # for row_idx, graph in enumerate(subgraphs):
    # #     print(f'graph n.{row_idx}: {graph}')

    # n_parts = 8
    # n_cuts, membership = pymetis.part_graph(n_parts, adjacency=grid.cells)
    # # n_cuts = 3
    # # membership = [1, 1, 1, 0, 1, 0, 0]

    # print(f'n_cells: {grid.get_num_cells()} (n_parts: {n_parts}) {(grid.get_num_cells() / n_parts) / grid.get_num_cells() * 100}')

    # for part in range(n_parts):
    #     cells_part = np.argwhere(np.array(membership) == part).ravel()
    #     print(f'len part n.{part}: {len(cells_part)}')

    #     # print(f'len: {len(cells_part)}) cells_part_{part}: {cells_part} = [', end='')
    #     # for cellid in cells_part:
    #     #     print(f'{grid.get_nodes_in_cell(cellid)}, ', end='')
    #     # print(']')

    # # nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel() # [3, 5, 6]
    # # nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel() # [0, 1, 2, 4]

    # # print(f'nodes_part_0: {nodes_part_0}')
    # # print(f'nodes_part_1: {nodes_part_1}')


    K = scipy.sparse.csr_matrix((np.zeros(len(row_idx)), (row_idx, col_idx)))
    f = np.zeros(len(row_idx))
    # Init element matrices
    ndofs_cell = dh.get_ndofs_per_cell()
    ke = np.zeros((ndofs_cell, ndofs_cell), dtype=np.float32)
    re = np.zeros(ndofs_cell, dtype=np.float32)
    # Init weak form handler
    weak_form = MomentumBalance(material=Elasticity(E=200e3, nu=0.3),
                                thickness=1.0,
                                element=CST())
    # Run K-assembly over the cells/elements
    for cellid in range(grid.get_num_cells()):
        ke.fill(0.0)
        re.fill(0.0)
        xe = grid.get_coordinates(cellid)
        dofs = dh.get_cell_dofs(cellid)
        ue = a[dofs] # NOTE: Not relevant as input for linear elasticity
        weak_form.run_element_routine(ke, re, xe, ue)
        for i, dof_i in enumerate(dofs):
            f[dof_i] += re[i]
            for j, dof_j in enumerate(dofs):
                K[dof_i, dof_j] += ke[i, j]
    # plt.spy(K)
    # # plt.savefig('sparse.png')
    # plt.show()
    # ==========================================================================
    # Solve system
    # ==========================================================================
    K_glob = K[free_dofs, :][:, free_dofs]
    # plt.spy(K_glob)
    # plt.savefig('sparse.png')
    # plt.show()
    f_glob = -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs]
    a[free_dofs] = scipy.sparse.linalg.spsolve(K_glob, f_glob)


def profile_solvers():

    def k_assembly(lcar=0.1):
        # ======================================================================
        # Define mesh
        # ======================================================================
        grid = generate_grid(lcar)
        dh = DofHandler(n_dofs_per_node=2, grid=grid)
        # ======================================================================
        # Define a and prescribe DoFs
        # ======================================================================
        a = np.zeros(dh.get_ndofs_total())
        bottom_dofs = dh.get_nodes_dofs(grid.nodesets['bottom'])
        left_dofs = dh.get_nodes_dofs(grid.nodesets['left'])
        top_dofs = dh.get_nodes_dofs(grid.nodesets['top'])
        prescribed_dofs = (bottom_dofs[:, 1], left_dofs[:, 0], top_dofs[:, 1])
        prescribed_dofs = np.concatenate(prescribed_dofs)
        free_dofs = np.setdiff1d(range(dh.get_ndofs_total()), prescribed_dofs)
        a[top_dofs[:, 1]] = -0.1
        # ======================================================================
        # Naive global K assembly
        # ======================================================================
        # Init global K matrix and f
        I, J = dh.get_sparsity_pattern()
        K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
        f = np.zeros(len(I))
        # Init element matrices
        ndofs_cell = dh.get_ndofs_per_cell()
        ke = np.zeros((ndofs_cell, ndofs_cell))
        re = np.zeros(ndofs_cell)
        xe = np.zeros((grid.get_num_nodes_per_cell(), 2))
        # Init weak form handler
        weak_form = MomentumBalance(material=Elasticity(E=200e3, nu=0.3),
                                    thickness=1.0,
                                    element=CST())
        # Run K-assembly over the cells/elements
        for cellid in range(grid.get_num_cells()):
            ke.fill(0.0)
            re.fill(0.0)
            xe = grid.get_coordinates(cellid)
            dofs = dh.get_cell_dofs(cellid)
            ue = a[dofs] # NOTE: Not relevant as input for linear elasticity
            weak_form.run_element_routine(ke, re, xe, ue)
            for (i, dof_i) in enumerate(dofs):
                for (j, dof_j) in enumerate(dofs):
                    K[dof_i, dof_j] += ke[i, j]
                f[dof_i] += re[i]
        return a, K, f, free_dofs, prescribed_dofs, grid

    n_runs = 20
    lcars = [0.5, 0.1, 0.05, 0.01]
    solvers = [
            ('spsolve', scipy.sparse.linalg.spsolve),
            ('bicg', scipy.sparse.linalg.bicg),
            ('bicgstab', scipy.sparse.linalg.bicgstab),
            ('cg', scipy.sparse.linalg.cg),
            ('cgs', scipy.sparse.linalg.cgs),
            ('gmres', scipy.sparse.linalg.gmres),
            ('lgmres', scipy.sparse.linalg.lgmres),
            ('minres', scipy.sparse.linalg.minres),
            ('qmr', scipy.sparse.linalg.qmr),
            ('gcrotmk', scipy.sparse.linalg.gcrotmk),
            ('tfqmr', scipy.sparse.linalg.tfqmr),
        ]
    with open('solvers_profiling.csv', 'w') as fp:
        fp.write(f'solver_name,lcar,n_nodes,n_cells,n_runs,cpu/gpu,t[s]\n')
        for lcar in lcars:
            a, K, f, free_dofs, prescribed_dofs, grid = k_assembly(lcar)
            A = K[free_dofs, :][:, free_dofs]
            b = -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs]
            n_nodes = len(grid.nodes)
            n_cells = len(grid.cells)
            for solver_name, solver in solvers:
                t = timeit.timeit(stmt=lambda: solver(A, b), number=n_runs)
                print(f'[Mesh size: {lcar}] {solver_name}: {t / n_runs:.4f} s')
                fp.write(f'{solver_name},{lcar},{n_nodes},{n_cells},{n_runs},cpu,{t / n_runs:.4f}\n')


import vedo

def plot_mesh():
    ug = vedo.UGrid('test.vtk')
    # print(ug.getArrayNames())
    # ug.selectCellArray('chem_0')
    vedo.show(ug, axes=True)

from collections import defaultdict


if __name__ == '__main__':
    # plot_mesh()
    # profile_solvers()
    test_run_simulation_plate_with_hole()