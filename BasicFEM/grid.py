import numpy as np
import pygmsh
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class Grid:
    # def __init__(self, nodes, cells, nodesets=None, bottom_nodes=None, left_nodes=None, top_nodes=None):
    def __init__(self, nodes, cells, nodesets=None):
        """Initialized Grid class.
        
        Models a grid structure.... Each node has a set of coordinates (either
        2D or 3D). Cells are a collection of three node IDs.
        
        Args:
            nodes (list): List of np.ndarray
            cells (list): List of Int Tuples, i.e. node IDs
            nodesets (dict): Dictionary of string to list of Ints (node numbers)
            bottom_nodes (list): List of indexes of the bottom nodes
            left_nodes (list): List of indexes of the left nodes
            top_nodes (list): List of indexes of the bottom nodes
        """
        self.nodes = np.array(nodes, dtype=np.float32)
        self.cells = np.array([np.int32(c) for c in cells], dtype=np.int32)
        self.nodesets = nodesets
        self.n_dim = self.nodes.shape[-1]
        # self.bottom_nodes = bottom_nodes
        # self.left_nodes = left_nodes
        # self.top_nodes = top_nodes

    def getcoordinates(self, cellid, xe=None):
        return self.get_coordinates(cellid, xe)
    
    def nnodes_per_cell(self):
        return len(self.cells[0])
    
    def get_coordinates(self, cellid, xe=None):
        """Given a cell ID, get the coordinates of the nodes in that cell.
        
        Args:
            xe (np.ndarray): List of elements coordinates, i.e. nodes defining
                a cell. Shape: (n_nodes_per_cell, n_dimensions)
            cellid (int): Progressive index ID of the cells in the model
        
        Returns:
            Tuple: List of elements coordinates of the specified cell
        """
        if xe is None:
            xe = np.zeros((self.nnodes_per_cell(), self.n_dim))
        nodes = self.cells[cellid]
        for (i, nodeid) in enumerate(nodes):
            xe[i] = self.nodes[nodeid]
        return xe

    def get_nodes_in_cell(self, cellid):
        return self.cells[cellid]

    def get_num_nodes_per_cell(self):
        return len(self.cells[0])

    def get_num_cells(self):
        return len(self.cells)

    def plot(self, nodal_values=None, filename=None, show=True):
        """
        Plots mesh and eventually nodal values. From:
        https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
        
        :param      nodal_values:  The nodal values
        :type       nodal_values:  np.array or list (must have same length as
                                   grid nodes)
        
        :returns:   None
        :rtype:     None
        """
        if len(self.nodes.shape) > 2:
            print(f'WARNING. Plotting for 3D nodes is not supported yet. Skipping plot generation.')
            return
        nodes_x = self.nodes[:, 0]
        nodes_y = self.nodes[:, 1]
        if nodal_values is not None:
            # Average nodal response per cell
            z = [nodal_values[cell].mean() for cell in self.cells]
            plt.tripcolor(nodes_x, nodes_y, triangles=self.cells, facecolors=z,
                          shading='flat', edgecolors='k')
            plt.colorbar()
        # Plot the finite element mesh
        for element in self.cells:
            x = [nodes_x[element[i]] for i in range(len(element))]
            y = [nodes_y[element[i]] for i in range(len(element))]
            plt.fill(x, y, edgecolor='black', fill=False)
        plt.axis('equal')
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()


class DofHandler:
    """Associated to each node: one node can have multiple DoF.
    
    DoF is a single "scalar" entity...
    
    Attributes:
        n_dofs_per_node (int): Description
    """
    def __init__(self, n_dofs_per_node, grid):
        """Initializes DofHandler class.
        
        Args:
            n_dofs_per_node (int): Description
        """
        self.n_dofs_per_node = n_dofs_per_node
        self.grid = grid
        self.cells_dofs = np.array([self.get_cell_dofs(c) for c in range(grid.get_num_cells())], dtype=np.int32)

    def ndofs_total(self, grid):
        n_nodes = grid.nodes.shape[0]
        return n_nodes * self.n_dofs_per_node
    
    def ndofs_per_cell(self, grid):
        return grid.nnodes_per_cell() * self.n_dofs_per_node

    def celldofs(self, dofs, grid, cellid):
        n = self.n_dofs_per_node
        for (i, nodeid) in enumerate(grid.cells[cellid]):
            for d in range(n):
                dofs[i * n + d] = nodeid * n + d
        return dofs
    
    def sparsity_pattern(self, grid):
        # TODO: Deprecated.
        ndofs_cell = self.ndofs_per_cell(grid)
        # dofs = [0 for i in range(ndofs_cell)]
        dofs = np.zeros(ndofs_cell, int)
        row_idx = []
        col_idx = []
        for cellid in range(len(grid.cells)):
            self.celldofs(dofs, grid, cellid)
            for dof in dofs:
                row_idx.extend(dofs)
                col_idx.extend([dof] * ndofs_cell)
        return row_idx, col_idx

    def getdofs(self, node_ids):
        # TODO: Deprecated.
        return self.get_dofs(node_ids)

    def get_dofs(self, node_ids):
        n = self.n_dofs_per_node
        dofs = np.empty((len(node_ids), n), dtype=np.int32)
        for i, nodeid in enumerate(node_ids):
            for d in range(n):
                dofs[i, d] = nodeid * n + d
        return dofs

    def get_ndofs_total(self):
        n_nodes = self.grid.nodes.shape[0]
        return n_nodes * self.n_dofs_per_node

    def get_ndofs_per_cell(self):
        return self.grid.nnodes_per_cell() * self.n_dofs_per_node

    def get_cell_dofs(self, cellid):
        dofs = np.empty(self.get_ndofs_per_cell(), dtype=np.int32)
        n_dofs = self.n_dofs_per_node
        for i, nodeid in enumerate(self.grid.cells[cellid]):
            for d in range(n_dofs):
                dofs[i * n_dofs + d] = nodeid * n_dofs + d
        return dofs

    def get_nodes_dofs(self, node_ids):
        n = self.n_dofs_per_node
        dofs = np.empty((len(node_ids), n), dtype=np.int32)
        for (i, nodeid) in enumerate(node_ids):
            for d in range(n):
                dofs[i, d] = nodeid * n + d
        return dofs

    def get_sparsity_pattern(self, return_shape=False):
        n_dofs_cell = self.get_ndofs_per_cell()
        row_idx, col_idx = [], []
        dim = -1
        for cellid in range(self.grid.get_num_cells()):
            dofs = self.get_cell_dofs(cellid)
            dim = max(dim, max(dofs) + 1)
            for dof in dofs:
                row_idx.extend(dofs)
                col_idx.extend([dof] * n_dofs_cell)
        if return_shape:
            return row_idx, col_idx, (dim, dim)
        else:
            return row_idx, col_idx

from scipy.spatial import Delaunay

def generate_grid(lcar=0.1):
    """
    Generate simple 2D grid
    
    :param      lcar:  characteristic length
    :type       lcar:  float
    
    :returns:   An instance of the Grid class
    :rtype:     Grid
    """
    with pygmsh.geo.Geometry() as geom:
        p0 = geom.add_point([0.0, 0.0], lcar)
        p1 = geom.add_point([1.0, 0.0], lcar)
        p2 = geom.add_point([2.0, 0.0], lcar)
        p3 = geom.add_point([2.0, 5.0], lcar)
        p4 = geom.add_point([0.0, 5.0], lcar)
        p5 = geom.add_point([0.0, 1.0], lcar)
        
        l0 = geom.add_line(p1, p2) 
        l1 = geom.add_line(p2, p3)
        l2 = geom.add_line(p3, p4)
        l3 = geom.add_line(p4, p5)
        
        ca1 = geom.add_circle_arc(p5, p0, p1)

        loop = geom.add_curve_loop([l0, l1, l2, l3, ca1])
        surface = geom.add_plane_surface(loop)
        geom.add_physical(l0, 'bottom')
        geom.add_physical(l2, 'top')
        geom.add_physical(l3, 'left')
        geom.add_physical(l1, 'right')
        geom.add_physical(ca1, 'hole')

        mesh = geom.generate_mesh(dim=2)

    # NOTE: The mesh always has 3 dimensions. Get only the first two coordinates
    nodes = mesh.points[:, :2]
    cells = mesh.cells_dict['triangle']

    # TODO: The origin is included in the mesh, but it shoudn't! See this Github
    # issue: https://github.com/meshpro/pygmsh/issues/562. The following is a
    # workaround to remove the origin node and decrease the nodeid of all cells
    # following it, i.e. the origin was at index 19, so once removed the other
    # nodes pointing at it in the cells need to be lowered by 1.
    # TODO: The following can be for sure vectorized in numpy...
    unconnected_nodes = []
    for i, node in enumerate(nodes):
        if np.allclose(node, 0):
            unconnected_nodes.append(i)
    for i in unconnected_nodes:
        nodes = np.delete(nodes, i, axis=0)
        cells[cells >= i] -= 1

    bottom_nodes = mesh.cell_sets_dict['bottom']['line']
    top_nodes = mesh.cell_sets_dict['top']['line']
    left_nodes = mesh.cell_sets_dict['left']['line']
    right_nodes = mesh.cell_sets_dict['right']['line']
    nodesets = {'bottom': bottom_nodes,
                'top': top_nodes,
                'left': left_nodes,
                'right': right_nodes
                }

    # mesh.write('test.vtk')
    basicfem_grid = Grid(nodes, cells, nodesets)
    return basicfem_grid
