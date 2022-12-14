import numpy as np
import pygmsh

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
        self.nodes = nodes
        self.cells = cells
        self.nodesets = nodesets
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
            xe = np.zeros((self.nnodes_per_cell(), 2))
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

    def ndofs_total(self, grid):
        nnodes = grid.nodes.shape[0]
        return nnodes * self.n_dofs_per_node
    
    def ndofs_per_cell(self, grid):
        return grid.nnodes_per_cell() * self.n_dofs_per_node

    def celldofs(self, dofs, grid, cellid):
        n = self.n_dofs_per_node
        for (i, nodeid) in enumerate(grid.cells[cellid]):
            for d in range(n):
                dofs[i * n + d] = nodeid * n + d
        return dofs
    
    def sparsity_pattern(self, grid):
        ndofs_cell = self.ndofs_per_cell(grid)
        # dofs = [0 for i in range(ndofs_cell)]
        dofs = np.zeros(ndofs_cell, int)
        I = []
        J = []
        for cellid in range(len(grid.cells)):
            self.celldofs(dofs, grid, cellid)
            for dof in dofs:
                I.extend(dofs)
                J.extend([dof] * ndofs_cell)
        return I, J

    def getdofs(self, node_ids):
        # TODO: Deprecated.
        return self.get_dofs(node_ids)

    def get_dofs(self, node_ids):
        n = self.n_dofs_per_node
        dofs = np.empty((len(node_ids), n), dtype=np.int32)
        for (i, nodeid) in enumerate(node_ids):
            for d in range(n):
                dofs[i, d] = nodeid * n + d
        return dofs

    def get_ndofs_total(self):
        nnodes = self.grid.nodes.shape[0]
        return nnodes * self.n_dofs_per_node

    def get_ndofs_per_cell(self):
        return self.grid.nnodes_per_cell() * self.n_dofs_per_node

    def get_cell_dofs(self, cellid):
        dofs = np.empty(self.get_ndofs_per_cell(), dtype=np.int32)
        n = self.n_dofs_per_node
        for (i, nodeid) in enumerate(self.grid.cells[cellid]):
            for d in range(n):
                dofs[i * n + d] = nodeid * n + d
        return dofs

    def get_nodes_dofs(self, node_ids):
        n = self.n_dofs_per_node
        dofs = np.empty((len(node_ids), n), dtype=np.int32)
        for (i, nodeid) in enumerate(node_ids):
            for d in range(n):
                dofs[i, d] = nodeid * n + d
        return dofs

    def get_sparsity_pattern(self):
        ndofs_cell = self.get_ndofs_per_cell()
        I, J = [], []
        for cellid in range(len(self.grid.cells)):
            dofs = self.get_cell_dofs(cellid)
            for dof in dofs:
                I.extend(dofs)
                J.extend([dof] * ndofs_cell)
        return I, J


def generate_grid(lcar=0.1):
    with pygmsh.geo.Geometry() as geom:
        p0 = geom.add_point([0.0, 0.0], lcar)
        p1 = geom.add_point([1.0, 0.0], lcar)
        p2 = geom.add_point([2.0, 0.0], lcar)
        p3 = geom.add_point([2.0, 2.0], lcar)
        p4 = geom.add_point([0.0, 2.0], lcar)
        p5 = geom.add_point([0.0, 1.0], lcar)
        
        l0 = geom.add_line(p1, p2) 
        l1 = geom.add_line(p2, p3)
        l2 = geom.add_line(p3, p4)
        l3 = geom.add_line(p4, p5)
        
        ca1 = geom.add_circle_arc(p5, p0, p1)

        loop = geom.add_curve_loop([l0, l1, l2, l3, ca1])
        surface = geom.add_plane_surface(loop)
        geom.add_physical(l0, 'bottom')
        geom.add_physical(l1, 'right')
        geom.add_physical(l2, 'top')
        geom.add_physical(l3, 'left')
        # geom.add_physical(ca1, 'hole')
        
        mesh = geom.generate_mesh()

    cells = mesh.cells_dict['triangle']
    nodes = mesh.points[:, 0:2]
    bottom_nodes = mesh.cell_sets_dict['bottom']['line']
    top_nodes = mesh.cell_sets_dict['top']['line']
    left_nodes = mesh.cell_sets_dict['left']['line']
    right_nodes = mesh.cell_sets_dict['right']['line']
    
    # with pygmsh.geo.Geometry() as geom:
    #     poly = geom.add_polygon(
    #         [
    #             [+0.0, +0.5],
    #             [-0.1, +0.1],
    #             [-0.5, +0.0],
    #             [-0.1, -0.1],
    #             [+0.0, -0.5],
    #             [+0.1, -0.1],
    #             [+0.5, +0.0],
    #             [+0.1, +0.1],
    #         ],
    #         mesh_size=0.05,
    #     )

    #     geom.twist(
    #         poly,
    #         translation_axis=[0, 0, 1],
    #         rotation_axis=[0, 0, 1],
    #         point_on_axis=[0, 0, 0],
    #         angle=np.pi / 3,
    #     )
    #     print('-' * 80)
    #     print(geom)
    #     print('-' * 80)
    #     mesh = geom.generate_mesh()
    # mesh.write('test.vtk')



    
    nodesets = {'bottom': bottom_nodes,
                'top': top_nodes,
                'left': left_nodes,
                'right': right_nodes
                }
    basicfem_grid = Grid(nodes, cells, nodesets)
    return basicfem_grid


# resolution = 0.01
# # Channel parameters
# L = 2.2
# H = 0.41
# c = [0.2, 0.2, 0]
# r = 0.05

# # Initialize empty geometry using the build in kernel in GMSH
# geometry = pygmsh.geo.Geometry()
# # Fetch model we would like to add data to
# model = geometry.__enter__()
# # Add circle
# circle = model.add_circle(c, r, mesh_size=resolution)

# # Add points with finer resolution on left side
# points = [model.add_point((0, 0, 0), mesh_size=resolution),
#           model.add_point((L, 0, 0), mesh_size=5*resolution),
#           model.add_point((L, H, 0), mesh_size=5*resolution),
#           model.add_point((0, H, 0), mesh_size=resolution)]

# # Add lines between all points creating the rectangle
# channel_lines = [model.add_line(points[i], points[i+1])
#                  for i in range(-1, len(points)-1)]

# # Create a line loop and plane surface for meshing
# channel_loop = model.add_curve_loop(channel_lines)
# plane_surface = model.add_plane_surface(channel_loop, holes=[circle.curve_loop])

# # Call gmsh kernel before add physical entities
# model.synchronize()

# model.add_physical([plane_surface], 'Volume')
# model.add_physical([channel_lines[0]], 'Inflow')
# model.add_physical([channel_lines[2]], 'Outflow')
# model.add_physical([channel_lines[1], channel_lines[3]], 'Walls')
# model.add_physical(circle.curve_loop.curves, 'Obstacle')
# mesh = geometry.generate_mesh(dim=2)