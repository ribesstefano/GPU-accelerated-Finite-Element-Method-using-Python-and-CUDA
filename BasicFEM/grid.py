import numpy as np

class Grid:
    def __init__(self, nodes, cells, nodesets):
        self.nodes = nodes # list of np.ndarray
        self.cells = cells # list of Int Tuples (node numbers)
        self.nodesets = nodesets # dict of string to list of Ints (node numbers)
    
    def getcoordinates(self, xe, cellid):
        nodes = self.cells[cellid]
        for (i, nodeid) in enumerate(nodes):
            xe[i] = self.nodes[nodeid]
        return xe
    
    def nnodes_per_cell(self):
        return len(self.cells[0])

class DofHandler:
    def __init__(self, ndofs_per_node):
        self.ndofs_per_node = ndofs_per_node

    def ndofs_total(self, grid):
        nnodes = grid.nodes.shape[0]
        return nnodes * self.ndofs_per_node
    
    def ndofs_per_cell(self, grid):
        return grid.nnodes_per_cell() * self.ndofs_per_node

    def celldofs(self, dofs, grid, cellid):
        n = self.ndofs_per_node
        for (i, nodeid) in enumerate(grid.cells[cellid]):
            for d in range(n):
                dofs[i*n+d] = nodeid * n + d
    
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
