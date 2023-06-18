import numpy as np
import random
from vessel_graph_generation.utilities import eukledian_dist
import warnings
from math import ceil
from typing import Tuple

GEOMETRY_SIZE = 76

class SimulationSpace:
    """
    The simulation space is a continous cuboid where the longest side length is exactly 1.
    It is possible to provide a geometry file to specify a custom shape inside the cuboid.
    """

    def __init__(self, config: dict, FAZ_center: np.ndarray = None, FAZ_radius: int = None):
        """
        Create a new Simulation space in wich the vessel trees are grown.

        Parameters:
        ----------
            - config: Simulation space configuration dictionary
            - FAZ_center: 2D simulation space position of the foveal avascular zone in the x-y plane
            - FAZ_radius: Radius of the foveal avascular zone
        """
        self.fixed_geometry = config.get('oxygen_sample_geometry_path') is not None

        # We voxelize the geometry space to efficiently sample oxygen sink positions
        if self.fixed_geometry:
            self.geometry: np.ndarray = np.load(config['oxygen_sample_geometry_path'])
            self.geometry_size = max(self.geometry.shape)
            self.shape = np.array(self.geometry.shape) / self.geometry_size
            self.size_x, self.size_y, self.size_z = self.shape
            self.valid_voxels = np.argwhere(self.geometry)
        else:
            self.size_x, self.size_y, self.size_z = config["no_voxel_x"], config["no_voxel_y"], config["no_voxel_z"]
            self.shape = np.array([self.size_x, self.size_y, self.size_z])
            assert all(self.shape>0), "The simulation space dimensions must be postive!"
            if any(self.shape>1) or all(self.shape!=1):
                warnings.warn("Warning: The largest dimension of the simulation space should be exactly one.")
            self.geometry_size = GEOMETRY_SIZE
            self.FAZ_center = np.array(FAZ_center)*self.geometry_size
            self.FAZ_radius = np.array(FAZ_radius)*self.geometry_size*0.5
            y_coords, x_coords = np.ogrid[:ceil(self.size_x*self.geometry_size), :ceil(self.size_y*self.geometry_size)]
            self.geometry = (x_coords - self.FAZ_center[0])**2 + (y_coords - self.FAZ_center[1])**2 > self.FAZ_radius**2
            self.geometry = np.expand_dims(self.geometry,-1)
            self.valid_voxels = np.argwhere(self.geometry) # Positions miss z dim


    def get_candidate_sinks(self, N: int) -> list:
        """
        Returns a list of random oxygen sink positions.

        Parameters:
        -----------
        - N: Number of candidate sinks
        """
        candidate_voxels = self.valid_voxels[np.random.randint(0,len(self.valid_voxels), N)]
        candidate_sinks = self._voxs_2_unit_poss(candidate_voxels)
        return list(filter(self.is_valid_position,candidate_sinks))
    
    def get_random_valid_position(self, along_axis: int, first=True):
        if self.fixed_geometry:
            ax_index = 0 if first else self.shape[along_axis]-1
            index_2d: list[int] = list(random.choice(np.argwhere(np.take(self.geometry, ax_index, axis=along_axis))))
            index_2d.insert(along_axis, ax_index)
            pos_3d = list(self._vox_2_unit_pos(index_2d))
            del pos_3d[along_axis]
            return pos_3d
        else:
            if along_axis == 0:
                return np.random.uniform(0,self.size_y), np.random.uniform(0,self.size_z)
            elif along_axis == 1:
                return np.random.uniform(0,self.size_x), np.random.uniform(0,self.size_z)
            else:
                index_2d: list[int] = self.valid_pixels[np.random.randint(0,len(self.valid_pixels))] + np.random.rand(2)
                index_2d.insert(along_axis, 0)
                pos_3d = list(self._vox_2_unit_pos(index_2d))
                del pos_3d[along_axis]
                return pos_3d
                
    def is_valid_position(self, pos: np.ndarray) -> bool:
        """
        Returns whether the given 3D position is within the simulation space and is considered valid
        """
        if any(pos>=self.shape) or any(pos<0):
            return False
        if self.fixed_geometry:
            return self.geometry[self._unit_pos_2_vox(pos)]>0
        else:
            return eukledian_dist(pos, self.FAZ_center)>self.FAZ_radius
    
    def _vox_2_unit_pos(self, index: Tuple[int,int,int]) -> Tuple[float, float, float]:
        """
        Converts a simulation space voxel postition into the unit cuboid coordinate system.
        """
        return tuple((np.array(index)+np.random.uniform(0,1,3)) / self.geometry_size)
    
    def _voxs_2_unit_poss(self, indices: np.ndarray) -> Tuple[float, float, float]:
        """
        Converts a list of simulation space voxel postitions into the unit cuboid coordinate system.
        """
        return (indices+np.random.uniform(0,1,(len(indices),3))) / self.geometry_size
    
    def _unit_pos_2_vox(self, pos: np.ndarray) -> tuple[int]:
        """
        Returns the simulation space voxel index of the given 3D position 
        """
        return tuple((pos*self.geometry_size).astype(np.uint16))
