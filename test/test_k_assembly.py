from BasicFEM import * 

import cProfile
import pstats
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import numba as nb

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
    K = scipy.sparse.csr_matrix((np.zeros(len(I)), (I, J)))
    K = np.zeros((a_len, f_len), dtype=np.float32)
    prescribed_dofs = np.concatenate((bottom_dofs[:, 1],
                                      left_dofs[:,0],
                                      top_dofs[:,1]))
    free_dofs = np.setdiff1d(range(dh.ndofs_total(grid)), prescribed_dofs)
    return K, f, grid, dh, a, weak_form


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
        xe = grid.getcoordinates(xe, cellid)
        dofs = dh.celldofs(dofs, grid, cellid)
        # print(f'Cell n.{cellid:3d}, DoF: {dofs}')
        ke.fill(0.0)
        re.fill(0.0)
        ue = a[dofs] # Not relevant as input for linear elasticity
        weak_form.element_routine(ke, re, element, xe, ue)
        # TODO(Kim): Probably there is a better way for doing this
        for (i, dof_i) in enumerate(dofs):
            for (j, dof_j) in enumerate(dofs):
                K[dof_i, dof_j] += ke[i, j]
            f[dof_i] += re[i]
    return K, f


def ref_code(grid):
    K, f, grid, dh, a, weak_form = setup_fem(grid)
    return k_assembly_ref(K, f, grid, dh, a, weak_form)


# @nb.jit(nopython=True, cache=True)
@nb.guvectorize('(float32[:,:,:], float32[:])', '(b,c,n)->(b)', cache=True, target='parallel')
def jacobi_det_cell3_2D(xe, detJ):
    for i in range(detJ.shape[0]):
        detJ[i] = ((xe[i, 2, 0] - xe[i, 1, 0]) * (xe[i, 3, 1] - xe[i, 1, 1]) -
                   (xe[i, 3, 0] - xe[i, 1, 0]) * (xe[i, 2, 1] - xe[i, 1, 1])
                  )


@nb.jit(nopython=True, cache=True)
def jacobian_cell3_2D(xe, J):
    # TODO(Kim): Not generally constant, could depend on iso_coord (xi)
    for i in range(xe.shape[0]):
        J[i, 0, 0] = xe[i, 2, 0] - xe[i, 1, 0]
        J[i, 0, 1] = xe[i, 3, 0] - xe[i, 1, 0]
        J[i, 1, 0] = xe[i, 2, 1] - xe[i, 1, 1]
        J[i, 1, 1] = xe[i, 3, 1] - xe[i, 1, 1]

# @nb.jit(nopython=True, cache=True)
# def shape_gradients(xe):
#     batch_size = xe.shape[0]
#     dNdxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32) # Shape: (3, 2)
#     detJ = np.zeros(batch_size, dtype=np.float32)
#     J = np.zeros((batch_size, 2, 2), dtype=np.float32) # Shape: (b, 2, 2)
#     dNdx = np.zeros((batch_size, 3, 2), dtype=np.float32) # Shape: (b, 3, 2)
#     jacobi_det_cell3_2D(xe, detJ)
#     jacobian_cell3_2D(xe, J)
#     # JinvT = np.linalg.inv(J).transpose((0, 2, 1))
#     JinvT = J.transpose((0, 2, 1))
#     for i in range(dNdx.shape[1]):
#         dNdx[:, i, :] = JinvT @ dNdxi[i]
#     return dNdx
#     # return np.einsum('bij,kj->bkj', JinvT, dNdxi) # Shape: (b, 3, 2)

def k_assembly_batched(K, f, grid, dh, a, weak_form):
    # TODO(Stefano): For now, it must be multiple of len(grid.cells)
    batch_size = 32
    n_cells = len(grid.cells)
    n_dims = 2
    # Init local variables for element loop (re-use them across iterations)
    n_dofs_cell = dh.ndofs_per_cell(grid)
    dofs = np.empty(n_dofs_cell, dtype=np.int32)
    ke = np.zeros((batch_size, n_dofs_cell, n_dofs_cell), dtype=np.float32)
    re = np.zeros((batch_size, n_dofs_cell), dtype=np.float32)
    xe = np.zeros((batch_size, grid.nnodes_per_cell(), n_dims), dtype=np.float32)
    ue = np.zeros((batch_size, n_dofs_cell), dtype=np.float32)
    element = CST()

    detJ = np.zeros(batch_size, dtype=np.float32)
    J = np.zeros((batch_size, 2, 2), dtype=np.float32) # Shape: (b, 2, 2)
    nqp = element.weights.shape[0]
    B = np.zeros((batch_size, nqp, 3, 6), dtype=np.float32)
    dNdxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32) # Shape: (3, 2)

    # dNdx = np.empty_like(dNdxi) # Shape: (b, 3, 2)
    # for i in range(dNdx.shape[0]): # dNdx.shape[0] == 3
    #     dNdx[i] = Jinv_t @ dNdxi[i]

    # Assemble K and f components
    for cellid in range(0, n_cells, batch_size):
        print(f'Working on cell n.{cellid}')
        # Assemble batched xe and ue
        for b in range(batch_size):
            xe[b] = grid.getcoordinates(xe[b], cellid + b)
            dofs = dh.celldofs(dofs, grid, cellid + b)
            ue[b] = a[dofs]

            ke.fill(0.0)
            re.fill(0.0)
            weak_form.element_routine(ke[b], re[b], element, xe[b], ue[b])
            # TODO(Kim): Probably there is a better way for doing this
            for (i, dof_i) in enumerate(dofs):
                for (j, dof_j) in enumerate(dofs):
                    K[dof_i, dof_j] += ke[b][i, j]
                f[dof_i] += re[b][i]


        # Get shape of gradients
        jacobi_det_cell3_2D(xe, detJ)
        jacobian_cell3_2D(xe, J)
        # JinvT = np.linalg.inv(J).transpose((0, 2, 1))
        JinvT = J.transpose((0, 2, 1))
        dNdx = np.einsum('bij,kj->bkj', JinvT, dNdxi) # Shape: (b, 3, 2)


        # ke.fill(0.0)
        # re.fill(0.0)
        # weak_form.element_routine(ke, re, element, xe, ue)
        # # TODO(Kim): Probably there is a better way for doing this
        # for (i, dof_i) in enumerate(dofs):
        #     for (j, dof_j) in enumerate(dofs):
        #         K[dof_i, dof_j] += ke[i, j]
        #     f[dof_i] += re[i]
    return K, f


def batched_code(grid):
    K, f, grid, dh, a, weak_form = setup_fem(grid)
    return k_assembly_batched(K, f, grid, dh, a, weak_form)


def test_k_assembly():
    pass
    # grid = generate_grid(0.5)
    # K, f = batched_code(grid)
    # K_ref, f_ref = ref_code(grid)
    # assert np.allclose(K, K_ref)
    # assert np.allclose(f, f_ref)


if __name__ == '__main__':
    test_k_assembly()