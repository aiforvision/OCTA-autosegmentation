import glob
import pickle
import os
import random
import math
from typing import Generator

import numpy as np

from vessel_graph_generation.arterial_tree import ArterialTree, Node
from vessel_graph_generation.simulation_space import SimulationSpace
from vessel_graph_generation.utilities import norm_vector
import sys
import csv

class Forest():

    def __init__(self, config, sim_space: SimulationSpace, arterial=True, template_forest: "Forest"=None):

        self.trees: list[ArterialTree] = []
        self.sim_space = sim_space
        self.simspace_size = sim_space.geometry.shape
        self.sim_scale = max(sim_space.geometry.shape)-1
        self.size = np.array(sim_space.geometry.shape) / max(sim_space.geometry.shape)
        self.size_x, self.size_y, self.size_z = tuple(self.size)
        self.arterial = arterial
        if template_forest is not None:
            self._initialize_tree_stumps_from_template_forest(template_forest)
        elif config['type'] == 'nerve':#hasattr(self.sim_space, "valid_start_voxels"):
            self._initialize_tree_stumps_from_simspace(config, config['d_0'], config['r_0'])
        elif config['type'] == 'stumps':
            self._initialize_tree_stumps(config, config['d_0'], config['r_0'])
        elif config['type'] == 'file':
            self._initialize_from_file(config, config['d_0'], config['r_0'])

    def _initialize_tree_stumps_from_template_forest(self, forest: "Forest"):
        for tree_counter, template_tree in enumerate(forest.get_trees()):
            tree_name = f'{"Arterial" if self.arterial else "Venous"}Tree{tree_counter}'
            nodes = list(template_tree.get_tree_iterator())
            wall_dim = np.where( (nodes[0].position==0) | (nodes[0].position == self.size) )[0]
            if len(wall_dim) > 0:
                wall_dim = wall_dim.item()
            else:
                wall_dim = random.choice([0,1,2])
                while nodes[0].position[wall_dim] == nodes[1].position[wall_dim]:
                    wall_dim = random.choice([0,1,2])
            node_pos = list(self.tree_pos_2_sim_vox(nodes[0].position))
            for i,p in enumerate(node_pos):
                node_pos[i] = p + random.choice(list(range(max(0-p, -self.sim_space.geometry.shape[i]//10), min(self.sim_space.geometry.shape[i]-p, self.sim_space.geometry.shape[i]//10+1)))) if p != 0 else p
            new_node_pos = list(self.sim_vox_2_tree_pos(node_pos))

            new_node_pos1 = [*new_node_pos]
            new_node_pos[wall_dim] = nodes[0].position[wall_dim]
            new_node_pos1[wall_dim] = nodes[1].position[wall_dim]
            tree = ArterialTree(tree_name, new_node_pos, nodes[0].radius, self.size_x, self.size_y, self.size_z, self)
            tree.add_node(new_node_pos1, radius=nodes[1].radius, parent=tree.root)
            self.trees.append(tree)

    def _initialize_tree_stumps_from_simspace(self, config, d_0: float, r_0: float):
        N_trees = config['N_trees']
        for tree_counter in range(N_trees):
            tree_name = f'{"Arterial" if self.arterial else "Venous"}Tree{tree_counter}'
            
            center = (0.43,0.88)
            radius = 0.01

            # random angle
            alpha = 2 * math.pi * random.random()
            # random radius
            r = radius * math.sqrt(random.random())
            # calculating coordinates
            x = r * math.cos(alpha) + center[0]
            y = r * math.sin(alpha) + center[1]
            z = random.random() * 1/76

            # indices = np.array(random.choice(self.sim_space.valid_start_voxels))
            # if len(indices)==2:
            #     indices = np.array([*indices, random.choice(range(self.sim_space.geometry.shape[-1]))])
            # tree_pos = self.sim_vox_2_tree_pos(indices)
            tree_pos = np.array([x,y,z])

            tree = ArterialTree(tree_name, tree_pos, r_0, self.size_x, self.size_y, self.size_z, self)
            # # Calculate most promising direction
            # lookahead = 10
            # direction = np.array([0,0,0])
            # for v in np.array([(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]):
            #     for i in range(1,lookahead+1):
            #         idx = indices + i*v
            #         if all(idx<self.sim_space.geometry.shape) and all(idx>=0):
            #             direction = direction + self.sim_space.geometry[tuple(idx)]*v
            # if all(direction == 0):
            #     direction = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)][np.random.choice(range(6))]
            # direction = norm_vector(direction) * d_0

            direction = norm_vector([random.random()-0.5, random.random()-0.5, 0]) * d_0
            
            tree.add_node(position=tree_pos+direction, radius=r_0, parent=tree.root)
            self.trees.append(tree)


    def _initialize_tree_stumps(self, config, d_0: float, r_0: float):

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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[0,:,:] > 0)))
                indices = random.choice(available_start_positions)
                y_position, z_position = self.sim_vox_2_tree_pos(indices)
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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[-1,:,:] > 0)))
                indices = random.choice(available_start_positions)
                y_position, z_position = self.sim_vox_2_tree_pos(indices)
                position = np.array([self.size_x,y_position,z_position])
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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[:,0,:] > 0)))
                indices = random.choice(available_start_positions)
                x_position, z_position = self.sim_vox_2_tree_pos(indices)
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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[:,-1,:] > 0)))
                indices = random.choice(available_start_positions)
                x_position, z_position = self.sim_vox_2_tree_pos(indices)
                position = np.array([x_position,self.size_y,z_position])
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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[:,:,0] > 0)))
                indices = random.choice(available_start_positions)
                x_position, y_position = self.sim_vox_2_tree_pos(indices)
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
                available_start_positions = list(zip(*np.where(self.sim_space.geometry[:,:,-1] > 0)))
                indices = random.choice(available_start_positions)
                x_position, y_position = self.sim_vox_2_tree_pos(indices)
                position = np.array([x_position,y_position,self.size_z])
                direction = np.array([
                    np.random.uniform(-1 if x_position-d_0>0 else 0, 1 if x_position+d_0<self.size_x else 0),
                    np.random.uniform(-1 if y_position-d_0>0 else 0, 1 if y_position+d_0<self.size_y else 0),
                    np.random.uniform(-1,-0.1)
                ])
                direction = direction / np.linalg.norm(direction) * d_0

                tree = ArterialTree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
                tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
                
                self.trees.append(tree)

    def sim_vox_2_tree_pos(self, index: tuple[int]) -> tuple[float]:
        return tuple([(i+random.uniform(0,1)) / self.sim_scale for i in index])

    def sim_voxs_2_tree_poss(self, candidate_voxels: np.ndarray) -> tuple[float]:
        return (candidate_voxels + np.random.uniform(0,1,candidate_voxels.shape)) / self.sim_scale

    def tree_pos_2_sim_vox(self, pos: tuple[float]) -> tuple[int]:
        return tuple([int(self.sim_scale * p) for p in pos])

    def is_inbounds(self, pos: np.ndarray) -> bool:
        return all(pos<self.size) and all(0<=pos) and self.sim_space.geometry[self.tree_pos_2_sim_vox(pos)]>0


    def _initialize_from_file(self, config):

        print(os.path.abspath(config['filepath']))

        tree_files = glob.glob(os.path.join(config['filepath'], '*.pkl'))

        for tree_file in tree_files:

            f = open(tree_file, 'rb')
            tree = pickle.load(f)
            f.close()

            self.trees.append(tree)

    
    def rescale(self, s=None, s_0=None, N_s=None, relative_rescale_factor=None):
        for tree in self.trees:
            tree.rescale(s, s_0, N_s, relative_rescale_factor)


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

    def save_pkl(self, save_directory='.'):
        name = f'{"Arterial" if self.arterial else "Venous"}Forest'
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, name + '.pkl')
        tmp = self.sim_space
        self.sim_space = None
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        self.simspace = tmp
