from BasicFEM import * 

import cProfile
import pstats
import numpy as cp
import scipy.sparse
import scipy.sparse.linalg
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import time
import timeit

def profile_gpu_solvers():
    dh = DofHandler(2)
    E = 200e3
    nu = 0.3
    material = Elasticity(E, nu)
    thickness = 1.0
    weak_form = MomentumBalance(material, thickness)
    element = CST()

    def k_assembly(lcar=0.1):
        grid = generate_grid(lcar)
        I, J = dh.sparsity_pattern(grid)
        K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
        f = np.zeros(len(I))

        ndofs_cell = dh.ndofs_per_cell(grid)
        ke = np.zeros((ndofs_cell, ndofs_cell))
        re = np.zeros(ndofs_cell)
        dofs = np.empty(ndofs_cell, np.int32)
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

        prescribed_dofs = (bottom_dofs[:, 1], left_dofs[:,0], top_dofs[:,1])
        prescribed_dofs = np.concatenate(prescribed_dofs)
        free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)
        return a, K, f, free_dofs, prescribed_dofs, grid

    n_runs = 20
    lcars = [0.5, 0.1, 0.05, 0.01]
    solvers = [
            ('spsolve', cupyx.scipy.sparse.linalg.spsolve),
            ('cg', cupyx.scipy.sparse.linalg.cg),
            ('cgs', cupyx.scipy.sparse.linalg.cgs),
            ('gmres', cupyx.scipy.sparse.linalg.gmres),
            ('minres', cupyx.scipy.sparse.linalg.minres),
        ]
    with open('solvers_profiling_gpu.csv', 'w') as fp:
        fp.write(f'solver_name,lcar,n_nodes,n_cells,n_runs,cpu/gpu,t[s]\n')
        for lcar in lcars:
            for solver_name, solver in solvers:
                a, K, f, free_dofs, prescribed_dofs, grid = k_assembly(lcar)
                A = cupyx.scipy.sparse.csr_matrix(K[free_dofs, :][:, free_dofs])
                b = -K[free_dofs, :][:, prescribed_dofs] @ a[prescribed_dofs]
                b = cp.asarray(b)
                print(f'running solver: {solver_name}')
                t = timeit.timeit(stmt=lambda: solver(A, b), number=n_runs)
                n_nodes = len(grid.nodes)
                n_cells = len(grid.cells)
                print(f'[Mesh size: {lcar}] {solver_name}: {t / n_runs:.4f} s')
                fp.write(f'{solver_name},{lcar},{n_nodes},{n_cells},{n_runs},gpu,{t / n_runs:.4f}\n')

if __name__ == '__main__':
    profile_gpu_solvers()