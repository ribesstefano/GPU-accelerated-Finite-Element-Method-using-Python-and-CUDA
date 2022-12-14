import numpy as np
from BasicFEM import *

def test_grid():
    """
    2 ---- 3
    | \\   |
    |   \\ |
    0 ---- 1
    """
    nodes = np.array([np.arange(4.0), np.arange(4.0)], dtype=np.float32).T
    cells = [(0,1,2), (1,3,2)]
    cells = np.array(cells, dtype=np.int32)
    grid = Grid(nodes, cells)
    xe = np.empty((3,2))
    grid.getcoordinates(1, xe)
    assert np.array_equal(xe, np.array([[1., 1.], [3., 3.], [2., 2.]]))

    # simple dof distribution
    dofhandler = DofHandler(2, grid)
    dofs = np.empty(6, np.int32)
    dofhandler.celldofs(dofs, grid, 1)
    assert np.array_equal(dofs, np.array([2,3,6,7,4,5]))