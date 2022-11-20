import numpy as np
import cupy as cp

class Grid:
    # def __init__(self, nodes, cells, nodesets=None, bottom_nodes=None, left_nodes=None, top_nodes=None):
    def __init__(self, nodes, cells, device='cpu'):
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
        if device == 'gpu':
            self.nodes = cp.asarray(nodes)
            self.cells = cp.asarray(cells)
        else:
            self.nodes = nodes
            self.cells = cells
        # self.nodesets = nodesets
        # self.bottom_nodes = bottom_nodes
        # self.left_nodes = left_nodes
        # self.top_nodes = top_nodes
    
    def getcoordinates(self, xe, cellid):
        """Given a cell ID, update the coordinates of the nodes in that cell.
        
        Args:
            xe (np.ndarray): List of elements coordinates, i.e. nodes defining
                a cell. Shape: (n_nodes_per_cell, n_dimensions)
            cellid (int): Progressive index ID of the cells in the model
        
        Returns:
            Tuple: List of elements coordinates of the specified cell
        """
        nodes = self.cells[cellid]
        for (i, nodeid) in enumerate(nodes):
            xe[i] = self.nodes[nodeid]
        return xe
    
    def nnodes_per_cell(self):
        return len(self.cells[0])

class DofHandler:
    """Associated to each node: one node can have multiple DoF.
    
    DoF is a single "scalar" entity...
    
    Attributes:
        ndofs_per_node (TYPE): Description
    """
    def __init__(self, ndofs_per_node, grid=None):
        """Initializes DofHandler class.
        
        Args:
            ndofs_per_node (TYPE): Description
        """
        self.ndofs_per_node = ndofs_per_node
        self.grid = grid

    def ndofs_total(self, grid):
        nnodes = grid.nodes.shape[0]
        return nnodes * self.ndofs_per_node
    
    def ndofs_per_cell(self, grid):
        return grid.nnodes_per_cell() * self.ndofs_per_node

    def celldofs(self, dofs, grid, cellid):
        n = self.ndofs_per_node
        for (i, nodeid) in enumerate(grid.cells[cellid]):
            for d in range(n):
                dofs[i * n + d] = nodeid * n + d
        return dofs
    
    def sparsity_pattern(self, grid):
        ndofs_cell = self.ndofs_per_cell(grid)
        dofs = [0 for i in range(ndofs_cell)]
        I = []
        J = []
        for cellid in range(len(grid.cells)):
            self.celldofs(dofs, grid, cellid)
            for dof in dofs:
                I.extend(dofs)
                J.extend([dof] * ndofs_cell)
        return I, J
    
    def getdofs(self, node_ids):
        n = self.ndofs_per_node
        dofs = np.empty((len(node_ids), n), int)
        for (i, nodeid) in enumerate(node_ids):
            for d in range(n):
                dofs[i, d] = nodeid * n + d
        return dofs  
