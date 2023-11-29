import os
import random
import math
from typing import Generator

import numpy as np

from vessel_graph_generation.arterial_tree import ArterialTree, Node
from vessel_graph_generation.simulation_space import SimulationSpace
from vessel_graph_generation.utilities import norm_vector
import csv

class Forest():

    def __init__(self, config: dict, d_0: float, r_0: float, sim_space: SimulationSpace, arterial=True, nerve_center: np.ndarray=None, nerve_radius: float=0):
        """
        Initialize a forest of multiple vessel trees.

        Parameters:
        ----------
            - config: forest configuration dictionary
            - d_0: Initial vessel length used for the root stumps.
            - r_0: Initial radius used for the root stumps.
            - sim_space: Simulation space in which the forest in grown.
            - arterial: If true, the forest initializes only arterial trees. If false, initializes only venous trees.
        """
        self.trees: list[ArterialTree] = []
        self.sim_space = sim_space
        self.size_x, self.size_y, self.size_z = self.sim_space.shape
        self.arterial = arterial
        if config['type'] == 'nerve':
            self._initialize_tree_stumps_from_nerve(config, d_0, r_0, nerve_center, nerve_radius)
        elif config['type'] == 'stumps':
            self._initialize_tree_stumps(config, d_0, r_0)
        else:
            raise NotImplementedError(f"The Forest initialization type '{config['type']}' is not implemented. Try 'stump' or 'nerve' instead.")

    def _initialize_tree_stumps_from_nerve(self, config, d_0: float, r_0: float, nerve_center: np.ndarray=None, nerve_radius: float=0):
        """
        Initialze the vessel network by placing all tree roots at the same position. By this, we replicate the optical nerve.
        This should only be used if you use a FOV that contains the optical nerve. Otherwise use 'stumps' initialization.

        Parameters:
        ----------
            - config: forest configuration dictionary
            - d_0: Initial vessel length used for the root stumps.
            - r_0: Initial radius used for the root stumps.
        """
        N_trees = config['N_trees']
        for tree_counter in range(N_trees):
            tree_name = f'{"Arterial" if self.arterial else "Venous"}Tree{tree_counter}'

            # random angle
            alpha = 2 * math.pi * random.random()
            # random radius
            r = nerve_radius * math.sqrt(random.random())
            # calculating coordinates
            x = r * math.cos(alpha) + nerve_center[1]
            y = r * math.sin(alpha) + nerve_center[0]
            z = random.random() * self.sim_space.size_z

            tree_pos = np.array([x,y,z])
            tree = ArterialTree(tree_name, tree_pos, r_0, self.size_x, self.size_y, self.size_z, self)
            direction = norm_vector([random.random()-0.5, random.random()-0.5, 0]) * d_0
            tree.add_node(position=tree_pos+direction, radius=r_0, parent=tree.root)
            self.trees.append(tree)

    def _initialize_tree_stumps(self, config, d_0: float, r_0: float):
        """
        Initialze the vessel network by placing all tree roots at the lateral faces of the simulation space cuboid.
        If your FOV is large enough to contain the optical nerve, you might want to use the 'nerve' initialization instead.

        Parameters:
        ----------
            - config: forest configuration dictionary
            - d_0: Initial vessel length used for the root stumps.
            - r_0: Initial radius used for the root stumps.
        """
        N_trees = config['N_trees']

        source_walls = []
        for key, entry in config['source_walls'].items():
            if entry:
                source_walls.append(key)

        tree_counter = 1
        for _ in range(N_trees):
            tree_name = f'{"Arterial" if self.arterial else "Venous"}Tree{tree_counter}'
            tree_counter += 1

            source_wall = random.choice(source_walls)

            if source_wall == 'x0':
                y_position, z_position = self.sim_space.get_random_valid_position(along_axis=0, first=True)
                position = np.array([0,y_position,z_position])
                direction = np.array([
                    np.random.uniform(0.1,1),
                    np.random.uniform(-1 if y_position-d_0>0 else 0, 1 if y_position+d_0<self.size_y else 0),
                    np.random.uniform(-1 if z_position-d_0>0 else 0, 1 if z_position+d_0<self.size_z else 0),
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

            elif source_wall == 'x1':
                y_position, z_position = self.sim_space.get_random_valid_position(along_axis=0, first=False)
                position = np.array([self.size_x-1e-6,y_position,z_position])
                direction = np.array([
                    np.random.uniform(-1,-0.1),
                    np.random.uniform(-1 if y_position-d_0>0 else 0, 1 if y_position+d_0<self.size_y else 0),
                    np.random.uniform(-1 if z_position-d_0>0 else 0, 1 if z_position+d_0<self.size_z else 0),
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

            elif source_wall == 'y0':
                x_position, z_position = self.sim_space.get_random_valid_position(along_axis=1, first=True)
                position = np.array([x_position,0,z_position])
                direction = np.array([
                    np.random.uniform(-1 if x_position-d_0>0 else 0, 1 if x_position+d_0<self.size_x else 0),
                    np.random.uniform(0.1,1),
                    np.random.uniform(-1 if z_position-d_0>0 else 0, 1 if z_position+d_0<self.size_z else 0),
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

            elif source_wall == 'y1':
                x_position, z_position = self.sim_space.get_random_valid_position(along_axis=1, first=False)
                position = np.array([x_position,self.size_y-1e-6,z_position])
                direction = np.array([
                    np.random.uniform(-1 if x_position-d_0>0 else 0, 1 if x_position+d_0<self.size_x else 0),
                    np.random.uniform(-1,-0.1),
                    np.random.uniform(-1 if z_position-d_0>0 else 0, 1 if z_position+d_0<self.size_z else 0),
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

            elif source_wall == 'z0':
                x_position, y_position = self.sim_space.get_random_valid_position(along_axis=2, first=True)
                position = np.array([x_position,y_position,0])
                direction = np.array([
                    np.random.uniform(-1 if x_position-d_0>0 else 0, 1 if x_position+d_0<self.size_x else 0),
                    np.random.uniform(-1 if y_position-d_0>0 else 0, 1 if y_position+d_0<self.size_y else 0),
                    np.random.uniform(0.1,1)
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

            elif source_wall == 'z1':
                x_position, y_position = self.sim_space.get_random_valid_position(along_axis=2, first=True)
                position = np.array([x_position,y_position,self.size_z-1e-6])
                direction = np.array([
                    np.random.uniform(-1 if x_position-d_0>0 else 0, 1 if x_position+d_0<self.size_x else 0),
                    np.random.uniform(-1 if y_position-d_0>0 else 0, 1 if y_position+d_0<self.size_y else 0),
                    np.random.uniform(-1,-0.1)
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

    def get_trees(self):
        return self.trees

    def get_nodes(self) -> Generator[Node, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node

    def get_node_coords(self) -> Generator[np.ndarray, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node.position

    def save(self, save_directory='.'):
        name = f'{"Arterial" if self.arterial else "Venous"}Forest'
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, name + '.csv')
        with open(filepath, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["node1", "node2", "radius"])
            for tree in self.get_trees():
                for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    proximal_node = current_node.get_proximal_node()
                    radius = current_node.radius
                    writer.writerow([current_node.position, proximal_node.position, radius])
