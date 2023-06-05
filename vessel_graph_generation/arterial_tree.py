from typing import Generator

import anytree
import numpy as np
from scipy.optimize import minimize


class Node(anytree.NodeMixin):

    def __init__(self, tree, name, position, radius, max_bounds=(np.inf, np.inf, np.inf), parent=None, children=None, kappa=4):

        super(Node, self).__init__()
        
        self.tree: ArterialTree = tree
        self.name = name
        
        self.position = np.array(position)
        self.radius = radius

        self.max_bounds = np.array(max_bounds)
        self.active = self._node_inbounds()

        self.kappa=kappa
        
        self.parent: Node = parent
        if children:
            self.children: list[Node] = children

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
        return self.tree.forest.is_inbounds(self.position)


    def detach(self):

        self.parent = None

    
    def optimize_branching_position(self):

        if self.is_root:
            raise RuntimeError('Unable to optimize branching position. Requested node is root.')
        if not self.is_bifurcation_node:
            raise RuntimeError('Unable to optimize branching position. Requested node is not a bifurcation node.')

        p_p = self.parent.position
        p_1 = self.children[0].position
        p_2 = self.children[1].position

        r_p = self.radius
        r_1 = self.children[0].radius
        r_2 = self.children[1].radius

        initial_guess = self.position
        
        def cumulative_distance_from_branch(p_b):
            return r_p ** 2 * np.linalg.norm(p_p - p_b) +\
                   r_1 ** 2 * np.linalg.norm(p_1 - p_b) +\
                   r_2 ** 2 * np.linalg.norm(p_2 - p_b)
        updated_position = minimize(cumulative_distance_from_branch, initial_guess, method='Nelder-Mead', tol=1e-15)['x']

        self.position = updated_position
        self.active = self._node_inbounds()


    def update_max_bounds(self, max_bounds):

        self.max_bounds = np.array(max_bounds)


    def update_position(self, position):

        self.position = position
        self.active = self._node_inbounds()


    def update_radius(self, radius):

        self.radius = radius


    def get_path_to_root_iterator(self):

        return reversed(self.path)


    def get_subtree_iterator(self, exclude_head=False):

        filter_ = lambda n: (n is not self if exclude_head else True)
        return anytree.LevelOrderIter(self, filter_)
    

    def get_distal_branch_length(self, child_index=None):

        if self.is_leaf:
            raise RuntimeError('Unable to analyze distal part. This node does not have any children.')
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                raise NotImplementedError('Oops.')
        elif self.is_inter_node or self.is_root:
            length = 0.0
            current_node = self
            while(current_node.is_inter_node):
                length += np.linalg.norm(current_node.children[0].position - current_node.position)
                current_node = current_node.children[0]
            
            return length


    def get_distal_node(self, child_index=None):

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
        elif self.is_bifurcation_node:
            if child_index is None:
                raise RuntimeError('Unable to analyze distal part. Unclear which branch to return.')
            else:
                return self.children[child_index].position - self.position
        elif self.is_inter_node or self.is_root:
            return self.children[0].position - self.position


    def get_proximal_bifurcation(self, one_below=False):
        
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            current_node = self
            while not (current_node.parent.is_root or current_node.parent.is_bifurcation_node):
                current_node = current_node.parent
            
            if one_below:
                return current_node
            else:
                return current_node.parent


    def get_proximal_branch_length(self):
        
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            length = 0.0
            current_node = self
            while current_node.is_leaf or current_node.is_inter_node:
                length += np.linalg.norm(current_node.position - current_node.parent.position)
                current_node = current_node.parent
            
            return length


    def get_proximal_node(self):
        
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            return self.parent


    def get_proximal_position(self):
        
        if self.is_root:
            raise RuntimeError('Unable to analyze proximal part. This node is the root.')
        else:
            return self.parent.position


    def get_proximal_radius(self):
        
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
        else:
            return self.position - self.parent.position

    def optimize_edge_radius_to_root(self):
        """
        Recursively adjusts the vessel radius of the parent edge according to
        Murray's law up to the root
        """
        if not self.is_root and not self.is_leaf:
            r_p = sum([child.radius**self.kappa for child in self.children])**(1/self.kappa)
            if self.radius == r_p:
                return
            self.update_radius(r_p)
            self.parent.optimize_edge_radius_to_root()
        


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
        self.root = Node(self, 'Root', position=root_position, radius=r_0, max_bounds=(size_x, size_y, size_z))

        self.name_counter = 1

    def __repr__(self):

        return str(anytree.RenderTree(self.root))


    def add_node(self, position, radius, parent, kappa=4) -> Node:

        new_name = 'Node' + str(self.name_counter)
        self.name_counter += 1

        return Node(self, new_name, position=position, radius=radius, max_bounds=(self.size_x, self.size_y, self.size_z), parent=parent, kappa=kappa)


    def add_existing_tree(self, tree, parent):

        for node in tree.get_tree_iterator(exclude_root=False):
            new_name = 'Node' + str(self.name_counter)
            self.name_counter += 1

            node.name = new_name
            node.tree = self

        tree.root.parent = parent


    def get_tree_iterator(self, exclude_root=False, only_active=False) -> Generator[Node, None, None]:

        filter_ = lambda n: (n.parent is not None if exclude_root else True) and (n.active if only_active else True)
        return anytree.LevelOrderIter(self.root, filter_)


    def get_tree_size(self, exclude_root=True, only_active=False):

        filter_ = lambda n: (n.parent is not None if exclude_root else True) and (n.active if only_active else True)
        return sum(1 for _ in anytree.LevelOrderIter(self.root, filter_))


    def rescale(self, s=None, s_0=None, N_s=None, relative_scaling_factor=None):
        if relative_scaling_factor is None:
            # Determine scaling factor
            previous_scaling_factor = self.scaling_factor
            self.scaling_factor = s_0 ** ((N_s - s) / (N_s - 1))
            relative_scaling_factor = self.scaling_factor / previous_scaling_factor

        self.size_x = self.scaling_factor * self.init_size_x
        self.size_y = self.scaling_factor * self.init_size_y
        self.size_z = self.scaling_factor * self.init_size_z

        self.r_0 = relative_scaling_factor * self.r_0

        for node in self.get_tree_iterator(exclude_root=False, only_active=False):

            node.update_max_bounds((self.size_x, self.size_y, self.size_z))
            node.update_position(relative_scaling_factor * node.position)
            node.update_radius(relative_scaling_factor * node.radius)
