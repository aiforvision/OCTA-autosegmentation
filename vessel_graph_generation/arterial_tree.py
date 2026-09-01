from collections import deque
from typing import Generator

import anytree
import numpy as np


class Node(anytree.NodeMixin):

    def __init__(self, tree, name, position, radius, parent=None, kappa=4, min_bifurcation_hops=0):
        """
        Creates a new node in the given tree.

        Parameters:
        -----------
            - tree: ArterialTree of which the node is part of
            - name: Name of the node
            - position: 3D position in the simulation space
            - radius: Radius of the proximal vessel segment
            - parent: Parent node. If None, the node is a root node.
            - kappa: The bifurcation exponent used to update the proximal vessel radius.
        """
        super(Node, self).__init__()
        
        self.tree: ArterialTree = tree
        self.name = name
        
        self.position = np.array(position)
        self.radius = radius

        self.active = self._node_inbounds()

        self.kappa=kappa
        self.min_bifurcation_hops: int = int(min_bifurcation_hops)
        
        self.parent: Node = parent

        # Shadow node attributes for multi-zone growth
        self.is_shadowed: bool = False  # True if node is shadowed in deeper layer
        self.shadow_pair: Node | None = None   # Reference to SVC node (if shadowed) or DVC node (if SVC)
        self.shadow_depth: float = None # Z-coordinate in DVC where shadowed

        # is_root and is_leaf are implemented by anytree
        self.is_inter_node = False
        self.is_bifurcation_node = False

        self.proximal_num_segments = -1
        current_node = self
        while current_node is not None:
            self.proximal_num_segments += 1
            current_node = current_node.parent


    def __repr__(self):
        return '{} (position: {}, radius: {}, active: {})'.format(self.name, self.position, self.radius, self.active)

    def _post_attach(self, parent):
        parent._update_node_status()


    def _post_detach(self, parent):
        parent._update_node_status()


    def _update_node_status(self):
        if self.parent is not None and len(self.children) == 1:
            self.is_inter_node = True
            self.is_bifurcation_node = False
        elif len(self.children) == 2:
            self.is_inter_node = False
            self.is_bifurcation_node = True
        else:
            self.is_inter_node = False
            self.is_bifurcation_node = False

    
    def _node_inbounds(self):
        return self.tree.forest.sim_space.is_valid_position(self.position)

    def update_radius(self, radius):
        self.radius = radius

    def get_distal_node(self, child_index=None):
        """
        Returns the indexed child node
        """
        if self.is_leaf:
            raise RuntimeError('Unable to analyze distal part. This node does not have any children.')
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                return self.children[child_index]
        elif self.is_inter_node or self.is_root:
            return self.children[0]


    def get_distal_position(self, child_index=None):
        """
        Returns the position of the indexed child node
        """
        if self.is_leaf:
            raise RuntimeError('Unable to analyze distal part. This node does not have any children.')
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                return self.children[child_index].position
        elif self.is_inter_node or self.is_root:
            return self.children[0].position


    def get_distal_radius(self, child_index=None):
        """
        Returns the radius of the vessel segment from to the given child node 
        """
        if self.is_leaf:
            raise RuntimeError('Unable to analyze distal part. This node does not have any children.')
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                return self.children[child_index].radius
        elif self.is_inter_node or self.is_root:
            return self.children[0].radius


    def get_distal_segment(self, child_index=None):
        """
        Returns direction vector from self to child with the given index
        """
        if self.is_leaf:
            raise RuntimeError('Unable to analyze distal part. This node does not have any children.')
        if self.is_shadowed:
            return self.parent.get_distal_segment()
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                return self.children[child_index].position - self.position
        elif self.is_inter_node or self.is_root:
            return self.children[0].position - self.position

    def get_proximal_node(self):
        """
        Returns the the parent node
        """
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            return self.parent


    def get_proximal_position(self):
        """
        Returns the position of the parent node
        """
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            return self.parent.position


    def get_proximal_radius(self):
        """
        Returns the radius of the vessel segment from parent to self 
        """
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            return self.radius


    def get_proximal_segment(self):
        """
        Returns direction vector from parent to self 
        """
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        elif self.is_shadowed:
            return self.parent.get_proximal_segment()
        else:
            return self.position - self.parent.position

    def optimize_edge_radius_to_root(self):
        """
        Recursively adjusts the vessel radius of the parent edge according to
        Murray's law up to the root
        """
        if not self.is_root and not self.is_leaf:
            kappa = max([child.kappa for child in self.children])
            r_p = sum([child.radius**kappa for child in self.children])**(1/kappa)
            if self.radius == r_p:
                return
            self.update_radius(r_p)
            self.parent.optimize_edge_radius_to_root()

    def graph_degree(self) -> int:
        """
        Return graph degree in the undirected vessel graph (parent + children).
        """
        return len(self.children) + (0 if self.is_root else 1)

    def get_neighbor_nodes(self) -> list['Node']:
        neighbors = list(self.children)
        if self.parent is not None:
            neighbors.append(self.parent)
        return neighbors

    def nodes_within_hops(self, hops: int, include_start: bool = False) -> set['Node']:
        if hops < 0:
            return set()
        visited = {self}
        queue = deque([(self, 0)])
        nodes = {self} if include_start else set()
        while queue:
            node, dist = queue.popleft()
            if dist >= hops:
                continue
            for neighbor in node.get_neighbor_nodes():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nodes.add(neighbor)
                queue.append((neighbor, dist + 1))
        return nodes

    def can_inter_node_sprout(self) -> bool:
        """
        Allow inter-node sprouting only if no nearby node (within hop budget)
        already has degree >= 3.
        """
        hops = int(getattr(self, "min_bifurcation_hops", 1))
        if hops <= 0:
            return True
        for near_node in self.nodes_within_hops(hops, include_start=False):
            if near_node.graph_degree() >= 3:
                return False
        return True

    def get_blocked_inter_nodes_after_bifurcation(self) -> list['Node']:
        """
        Return degree-2 neighbors within hop budget that should be removed from
        active sprouting candidates after this node became degree-3.
        """
        if self.graph_degree() < 3:
            return []
        hops = int(getattr(self, "min_bifurcation_hops", 1))
        if hops <= 0:
            return []
        return [
            node
            for node in self.nodes_within_hops(hops, include_start=False)
            if node.graph_degree() == 2
        ]
    
    def get_degree(self) -> int:
        """
        Return the number of children this node has (0, 1, or 2).
        """
        return len(self.children)
    
    def create_shadow(self, shadow_depth: float) -> 'Node':
        """
        Create a shadowed copy of this node at specified depth in DVC layer.
        
        Parameters:
        -----------
        - shadow_depth: Z-coordinate in DVC where shadow should be placed
        
        Returns:
        --------
        - Shadow node at (x, y, shadow_depth)
        """
        shadow_pos = np.array([self.position[0], self.position[1], shadow_depth])
        shadow_name = f"{self.name}_shadow"
        
        # Create shadow node with same parent structure
        shadow_node = Node(
            tree=self.tree,
            name=shadow_name,
            position=shadow_pos,
            radius=self.tree.r_0,  # Start with root radius
            parent=self,  # Keep same parent for tree structure
            kappa=self.kappa
        )
        
        shadow_node.is_shadowed = True
        shadow_node.shadow_depth = shadow_depth
        
        return shadow_node

class ArterialTree():
    
    def __init__(self, name, root_position, r_0, size_x, size_y, size_z, forest):

        super().__init__()

        self.name = name

        self.init_size_x = size_x
        self.init_size_y = size_y
        self.init_size_z = size_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        
        self.r_0 = r_0

        self.scaling_factor = 1.0

        self.forest = forest
        self.root = Node(self, 'Root', position=root_position, radius=r_0)

        self.name_counter = 1

    def __repr__(self):

        return str(anytree.RenderTree(self.root))


    def add_node(self, position, radius, parent, kappa=4, min_bifurcation_hops=1) -> Node:

        new_name = 'Node' + str(self.name_counter)
        self.name_counter += 1

        return Node(
            self,
            new_name,
            position=position,
            radius=radius,
            parent=parent,
            kappa=kappa,
            min_bifurcation_hops=min_bifurcation_hops,
        )


    def get_tree_iterator(self, exclude_root=False, only_active=False) -> Generator[Node, None, None]:
        def filter_(n: Node) -> bool:
            return (n.parent is not None if exclude_root else True) and (n.active if only_active else True) and (not n.is_shadowed)
        return anytree.LevelOrderIter(self.root, filter_)
