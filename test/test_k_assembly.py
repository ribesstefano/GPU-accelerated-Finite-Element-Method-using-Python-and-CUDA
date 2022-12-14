from BasicFEM import * 

import cProfile
import pstats
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import numba as nb
import time

def setup_fem(grid):
    n_dims = 2
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
                                      left_dofs[:,0],
                                      top_dofs[:,1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)
    return K.copy(), f.copy(), dh, a, weak_form


def k_assembly_ref(K, f, grid, dh, a, weak_form):
    n_dims = 2
    # Init local variables for element loop (re-use them across iterations)
    n_dofs_cell = dh.ndofs_per_cell(grid)
    dofs = np.empty(n_dofs_cell, dtype=np.int32)
    ke = np.zeros((n_dofs_cell, n_dofs_cell), dtype=np.float32)
    re = np.zeros(n_dofs_cell, dtype=np.float32)
    xe = np.zeros((grid.nnodes_per_cell(), n_dims), dtype=np.float32)
    element = CST()
    # Assemble K and f components
    for cellid in range(len(grid.cells)):
        # NOTE(Stefano): The methods getcoordinates and celldofs now return the
        # updated values. Makes the code more readable and Pythonic. Also, it
        # doesn't affect performance (it's just passing around references, i.e.
        # pointers, not the actual values).
        xe = grid.getcoordinates(cellid, xe)
        dofs = dh.celldofs(dofs, grid, cellid)
        # print(f'Cell n.{cellid:3d}, DoF: {dofs}')
        ke.fill(0.0)
        re.fill(0.0)
        ue = a[dofs] # Not relevant as input for linear elasticity
        weak_form.element_routine(ke, re, element, xe, ue)
        # # TODO(Kim): Probably there is a better way for doing this
        # for (i, dof_i) in enumerate(dofs):
        #     for (j, dof_j) in enumerate(dofs):
        #         K[dof_i, dof_j] += ke[i, j]
        #     f[dof_i] += re[i]
    return K, f


def ref_code(grid):
    K, f, dh, a, weak_form = setup_fem(grid)
    start_time = time.time()
    k_assembly_ref(K, f, grid, dh, a, weak_form)
    end_time = time.time()
    print(f'Reference code time: {end_time - start_time:.4f} s')
    return K, f


# @nb.jit(nopython=True, cache=True)
@nb.guvectorize('(float32[:,:,:], float32[:])', '(b,c,n)->(b)', cache=True, target='parallel')
def jacobi_det_cell3_2D(xe, detJ):
    for i in range(detJ.shape[0]):
        detJ[i] = ((xe[i, 1, 0] - xe[i, 0, 0]) * (xe[i, 2, 1] - xe[i, 0, 1]) -
                   (xe[i, 2, 0] - xe[i, 0, 0]) * (xe[i, 1, 1] - xe[i, 0, 1])
                  )


@nb.jit(nopython=True, cache=True)
def jacobian_cell3_2D(xe, J):
    # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
    for i in range(xe.shape[0]):
        J[i, 0, 0] = xe[i, 1, 0] - xe[i, 0, 0]
        J[i, 0, 1] = xe[i, 2, 0] - xe[i, 0, 0]
        J[i, 1, 0] = xe[i, 1, 1] - xe[i, 0, 1]
        J[i, 1, 1] = xe[i, 2, 1] - xe[i, 0, 1]


@nb.jit(nopython=True, cache=True)
def assemble_B(dNdx, B):
    for i in range(dNdx.shape[0]):
        for j in range(dNdx.shape[1]):
            dNidx = dNdx[i, j, 0]
            dNidy = dNdx[i, j, 1]
            B[i, 0, 2 * j] = dNidx
            B[i, 2, 2 * j] = dNidy
            B[i, 1, 2 * j + 1] = dNidy
            B[i, 2, 2 * j + 1] = dNidx
    return B


@nb.guvectorize('(float32[:,:], float32[:], float32[:,:])', '(b,c),(b)->(b,c)', cache=True, target='parallel')
def accum_re(BTsigma, detJw, re):
    for i in range(BTsigma.shape[0]):
        for j in range(BTsigma.shape[1]):
            re[i, j] = BTsigma[i, j] * detJw[i]


@nb.guvectorize('(float32[:,:,:], float32[:], float32[:,:,:])', '(b,c,d),(b)->(b,c,d)', cache=True, target='parallel')
def accum_ke(BTdsdeB, detJw, ke):
    for i in range(BTdsdeB.shape[0]):
        for j in range(BTdsdeB.shape[1]):
            for k in range(BTdsdeB.shape[2]):
                ke[i, j, k] = BTdsdeB[i, j, k] * detJw[i]


def k_assembly_batched(K, f, grid, dh, a, weak_form, batch_size=32):
    # TODO(Stefano): For now, it must be multiple of len(grid.cells)
    n_cells = len(grid.cells)
    n_dims = 2
    batch_sz = min(batch_size, n_cells)
    # Init local variables for element loop (re-use them across iterations)
    n_dofs_cell = dh.ndofs_per_cell(grid)
    dofs = np.empty((batch_sz, n_dofs_cell), dtype=np.int32)
    ke = np.zeros((batch_sz, n_dofs_cell, n_dofs_cell), dtype=np.float32)
    re = np.zeros((batch_sz, n_dofs_cell), dtype=np.float32)
    xe = np.zeros((batch_sz, grid.nnodes_per_cell(), n_dims), dtype=np.float32)
    ue = np.zeros((batch_sz, n_dofs_cell), dtype=np.float32)
    element = CST()

    detJ = np.zeros(batch_sz, dtype=np.float32)
    J = np.zeros((batch_sz, 2, 2), dtype=np.float32) # Shape: (b, 2, 2)
    nqp = element.weights.shape[0]
    # TODO(Stefano): For now, B doesn't depend on nqp
    B = np.zeros((batch_sz, grid.nnodes_per_cell(), n_dofs_cell), dtype=np.float32)
    dNdxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32) # Shape: (3, 2)
    dNdx = np.empty((batch_sz, *dNdxi.shape), dtype=np.float32)


    # @nb.jit(nopython=True, cache=True)
    def element_routine(stiffness, thickness, ke, re, weights, xe, ue, detJ, J, B, dNdxi, dNdx):
        # ke.fill(0.0)
        # re.fill(0.0)
        # Get shape of gradients
        jacobi_det_cell3_2D(xe, detJ)
        # assert np.allclose(detJ[-1], element.jacobi_determinant(xe[-1]))
        jacobian_cell3_2D(xe, J)
        # assert np.allclose(J[-1], element.jacobian(xe[-1]))
        JinvT = np.linalg.inv(J).transpose((0, 2, 1))
        # assert np.allclose(JinvT[-1], np.linalg.inv(element.jacobian(xe[-1])).T)
        dNdx = np.matmul(JinvT, dNdxi.T).transpose((0, 2, 1))
        # assert np.allclose(dNdx[-1], element.shape_gradients(xe[-1]))
        # Get B
        assemble_B(dNdx, B)
        # assert np.allclose(B[-1], element.B_operator(xe[-1]))
        BT = B.transpose((0, 2, 1))
        # Setup element routine
        detJw = detJ * weights
        dsde = stiffness
        # Run element routine
        epsilon = np.einsum('bij,bj->bi', B, ue)
        # assert np.allclose(epsilon[-1], element.B_operator(xe[-1]) @ ue[-1])
        sigma = np.einsum('ij,bj->bi', dsde, epsilon)
        BTsigma = np.einsum('bij,bj->bi', BT, sigma)
        BTdsdeB = np.matmul(np.matmul(BT, dsde), B)
        accum_re(BTsigma, detJw, re)
        accum_ke(BTdsdeB, detJw, ke)
        re *= thickness
        ke *= thickness


    # Initialize profiler
    profiler = cProfile.Profile()
    # Start profiling
    profiler.enable()

    # Assemble K and f components
    for cellid in range(0, n_cells, batch_sz):
        actual_batch_sz = min(batch_sz, n_cells - cellid)
        # print(f'Working on cell n.{cellid} (batch size: {actual_batch_sz}, n_cells: {n_cells})')
        # Assemble batched xe and ue
        for b in range(actual_batch_sz):
            xe[b] = grid.getcoordinates(cellid + b, xe[b])
            dofs[b] = dh.celldofs(dofs[b], grid, cellid + b)
            ue[b] = a[dofs[b]]
        element_routine(weak_form.material.stiffness, weak_form.thickness, ke, re, element.weights, xe, ue, detJ, J, B, dNdxi, dNdx)
        # # TODO(Kim): Probably there is a better way for doing this
        # for b in range(actual_batch_sz):
        #     for (i, dof_i) in enumerate(dofs[b]):
        #         for (j, dof_j) in enumerate(dofs[b]):
        #             K[dof_i, dof_j] += ke[b, i, j]
        #         f[dof_i] += re[b, i]
    # Stop profiling
    profiler.disable()
    # Create statistics from the profiler, sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # Print the 10 (inclusive) most expensive functions
    stats.print_stats(100)
    # stats.dump_stats('../program.prof')
    return K, f


def batched_code(grid):
    print(f'Number of cells: {len(grid.cells)}')
    K, f, dh, a, weak_form = setup_fem(grid)
    start_time = time.time()
    k_assembly_batched(K, f, grid, dh, a, weak_form, batch_size=128)
    end_time = time.time()
    print(f'Batched code time: {end_time - start_time:.4f} s')
    # Run it twice to remove Numba combilation time
    K, f, dh, a, weak_form = setup_fem(grid)
    start_time = time.time()
    k_assembly_batched(K, f, grid, dh, a, weak_form, batch_size=128)
    end_time = time.time()
    print(f'Batched code time: {end_time - start_time:.4f} s')
    return K, f


def test_k_assembly():
    # grid = generate_grid(0.5)
    grid = generate_grid(0.06)
    K_ref, f_ref = ref_code(grid)
    K, f = batched_code(grid)
    assert np.allclose(K, K_ref)
    assert np.allclose(f, f_ref)


if __name__ == '__main__':
    test_k_assembly()