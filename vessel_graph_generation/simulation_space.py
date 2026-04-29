import warnings
from math import ceil
from typing import Tuple

import numpy as np
from scipy.ndimage import convolve

from vessel_graph_generation.utilities import eukledian_dist

GEOMETRY_SIZE = 76
# Prefer stump placements nearer the nominal side: see ``stump_surface_depth_slices`` and
# ``stump_surface_proximity_bonus`` on ``SimulationSpace`` sampling weights.
DEFAULT_STUMP_SURFACE_DEPTH_SLICES = 5
DEFAULT_STUMP_SURFACE_PROXIMITY_BONUS = 1.0

class SimulationSpace:
    """
    The simulation space is a continous cuboid where the longest side length is exactly 1.
    It is possible to provide a geometry file to specify a custom shape inside the cuboid.
    """

    def __init__(self, config: dict, FAZ_center: np.ndarray = None, FAZ_radius: int = None, nerve_center: np.ndarray = None, nerve_radius: float=None):
        """
        Create a new Simulation space in wich the vessel trees are grown.

        Parameters:
        ----------
            - config: Simulation space configuration dictionary. Optional keys include
              ``stump_surface_depth_slices`` (int, default 5) and
              ``stump_surface_proximity_bonus`` (float, default 1): sampling weights multiply by up to
              ``1 + bonus`` when the surface voxel lies on the boundary face, decaying linearly to
              ``1`` when its index along that axis is at least ``stump_surface_depth_slices`` from
              the side.
            - FAZ_center: 2D simulation space position of the foveal avascular zone in the x-y plane
            - FAZ_radius: Radius of the foveal avascular zone
        """
        self.fixed_geometry = config.get('oxygen_sample_geometry_path') is not None
        self._stump_surface_depth_slices = int(
            config.get("stump_surface_depth_slices", DEFAULT_STUMP_SURFACE_DEPTH_SLICES)
        )
        self._stump_surface_proximity_bonus = float(
            config.get("stump_surface_proximity_bonus", DEFAULT_STUMP_SURFACE_PROXIMITY_BONUS)
        )

        # We voxelize the geometry space to efficiently sample oxygen sink positions
        if self.fixed_geometry:
            if config['oxygen_sample_geometry_path'].endswith('.npy'):
                raw_geometry: np.ndarray = np.load(config['oxygen_sample_geometry_path'])
            elif config['oxygen_sample_geometry_path'].endswith('.nii.gz'):
                import nibabel as nib
                raw_geometry: np.ndarray = nib.load(config['oxygen_sample_geometry_path']).get_fdata()
            
            self.geometry_size = max(raw_geometry.shape)
            self.shape = np.array(raw_geometry.shape) / self.geometry_size
            self.size_x, self.size_y, self.size_z = self.shape
            
            # Detect if multi-zone geometry or binary mask
            self.zones, self.multi_zone = self._detect_zones(raw_geometry, config.get('zone_mapping'))
            
            if self.multi_zone:
                # Store full geometry and initialize with first zone
                self.full_geometry = raw_geometry
                self.current_zone = None
                # Set active zone to the first one (will be set properly in greenhouse)
                first_zone_name = list(self.zones.keys())[0]
                self.set_active_zone(first_zone_name)
            else:
                # Legacy binary mask behavior
                self.geometry = raw_geometry.astype(bool)
                self.valid_voxels = np.argwhere(self.geometry)
                self.current_zone = None
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
            if all(nerve_center-nerve_radius<=1):
                self.nerve_center = np.array(nerve_center)*self.geometry_size
                self.nerve_radius = np.array(nerve_radius)*self.geometry_size
                self.geometry &= (x_coords - self.nerve_center[0])**2 + (y_coords - self.nerve_center[1])**2 > self.nerve_radius**2
            else:
                self.nerve_radius=None
                self.nerve_center=None
            self.geometry = np.expand_dims(self.geometry,-1)
            self.valid_voxels = np.argwhere(self.geometry) # Positions miss z dim
            self.multi_zone = False
            self.zones = {}

    def _detect_zones(self, geometry: np.ndarray, zone_mapping: dict = None) -> tuple[dict, bool]:
        """
        Automatically detect unique non-zero values in geometry and map them to zones.
        
        Parameters:
        -----------
        - geometry: The loaded geometry array
        - zone_mapping: Optional explicit mapping of zone names to values (e.g., {"SVC": 255, "DVC": 128})
        
        Returns:
        --------
        - zones: Dictionary mapping zone names to their properties
        - multi_zone: Boolean indicating if multiple zones were detected
        """
        unique_values = np.unique(geometry)
        non_zero_values = unique_values[unique_values > 0]
        
        # Check if this is a binary mask or multi-zone
        if len(non_zero_values) == 1:
            # Binary mask - maintain legacy behavior
            return {}, False
        
        # Multiple zones detected
        zones = {}
        
        if zone_mapping:
            # Use explicit mapping
            for zone_name, zone_value in zone_mapping.items():
                if zone_value in non_zero_values:
                    zones[zone_name] = self._create_zone_info(geometry, zone_value)
        else:
            # Auto-detect: sort values and assign names based on position
            # Assuming higher values = superficial layers, lower values = deeper layers
            sorted_values = sorted(non_zero_values, reverse=True)
            zone_names = ["SVC", "DVC", "ICC", "IVC"]  # Common retinal layers
            
            for i, value in enumerate(sorted_values[:len(zone_names)]):
                zone_name = zone_names[i] if i < len(zone_names) else f"Zone{i}"
                zones[zone_name] = self._create_zone_info(geometry, value)
        
        return zones, True
    
    def _create_zone_info(self, geometry: np.ndarray, zone_value: int) -> dict:
        """
        Extract zone-specific information from geometry.
        
        Returns a dictionary with zone properties.
        """
        zone_mask = (geometry == zone_value)
        zone_voxels = np.argwhere(zone_mask)
        
        return {
            'value': zone_value,
            'mask': zone_mask,
            'voxels': zone_voxels,
        }
    
    def set_active_zone(self, zone_name: str):
        """
        Switch simulation space to use specific zone geometry.
        
        Parameters:
        -----------
        - zone_name: Name of the zone to activate (e.g., "SVC", "DVC")
        """
        if not self.multi_zone:
            warnings.warn("set_active_zone called but geometry is not multi-zone. Ignoring.")
            return
        
        if zone_name not in self.zones:
            raise ValueError(f"Zone '{zone_name}' not found. Available zones: {list(self.zones.keys())}")
        
        self.current_zone = zone_name
        zone_info = self.zones[zone_name]
        self.geometry = zone_info['mask']
        self.valid_voxels = zone_info['voxels']
    
    def get_zone_height_range(self, zone_name: str, x: float, y: float) -> tuple[float, float]:
        """
        Get min and max z-coordinates for a zone in unit coordinates.
        
        Parameters:
        -----------
        - zone_name: Name of the zone
        - x: x-coordinate in unit space
        - y: y-coordinate in unit space
        
        Returns:
        --------
        - (z_min, z_max): Tuple of minimum and maximum z-coordinates
        """
        if not self.multi_zone or zone_name not in self.zones:
            raise ValueError(f"Zone '{zone_name}' not found or geometry is not multi-zone")
        zone_mask = self.zones[zone_name]["mask"]

        if not (0 <= x <= 1) or not (0 <= y <= 1):
            return -1, -1
        
        x_idx = int(x * self.geometry_size)
        y_idx = int(y * self.geometry_size)
        z_indices = np.where(zone_mask[x_idx, y_idx, :])[0]
        if len(z_indices) == 0:
            return -1, -1
        z_min = z_indices.min() / self.geometry_size
        z_max = z_indices.max() / self.geometry_size
        return z_min, z_max
        

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

    def _neighbor_counts_26(self) -> np.ndarray:
        """Count True voxels in the 26-neighborhood (excluding center) per voxel."""
        g = self.geometry.astype(np.float64)
        kernel = np.ones((3, 3, 3), dtype=np.float64)
        return convolve(g, kernel, mode="constant") - g

    def _surface_voxel_indices(self, along_axis: int, first: bool, depth: int) -> np.ndarray:
        """
        Outer-surface voxels within `depth` slices from the chosen side.

        For `first=True`, only columns where geometry is False for all indices closer to 0 along
        `along_axis` than the voxel are kept (first hit when marching from the boundary inward).
        """
        g = np.asarray(self.geometry, dtype=bool)
        L = int(g.shape[along_axis])
        if L == 0:
            return np.zeros((0, 3), dtype=np.intp)
        d = max(0, min(int(depth), L))
        if d == 0:
            return np.zeros((0, 3), dtype=np.intp)

        sl = [slice(None)] * 3
        if first:
            sl[along_axis] = slice(0, d)
        else:
            sl[along_axis] = slice(L - d, L)
        sub = g[tuple(sl)]
        mv = np.moveaxis(sub, along_axis, -1)
        if not first:
            mv = np.flip(mv, axis=-1)

        first_idx = np.argmax(mv, axis=-1)
        hit = np.take_along_axis(mv, first_idx[..., np.newaxis], axis=-1)[..., 0]
        valid = mv.any(axis=-1) & hit

        ii = np.argwhere(valid)
        if ii.size == 0:
            return np.zeros((0, 3), dtype=np.intp)

        full = np.zeros((ii.shape[0], 3), dtype=np.intp)
        other_axes = [ax for ax in (0, 1, 2) if ax != along_axis]
        for k, ax in enumerate(other_axes):
            full[:, ax] = ii[:, k]
        if first:
            full[:, along_axis] = first_idx[tuple(ii.T)]
        else:
            full[:, along_axis] = L - 1 - first_idx[tuple(ii.T)]

        return full

    def get_random_valid_position(self, along_axis: int, first=True):
        if self.fixed_geometry:
            axis_len = int(self.geometry.shape[along_axis])
            indices = self._surface_voxel_indices(along_axis, first, axis_len)
            if len(indices) == 0:
                side = "low" if first else "high"
                raise ValueError(
                    f"No valid outer-surface voxels along axis {along_axis} ({side} side); "
                    "check the geometry mask."
                )

            neighbors = self._neighbor_counts_26()
            w = neighbors[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float64)
            depth_ref = float(max(self._stump_surface_depth_slices, 1))
            coord_along = indices[:, along_axis].astype(np.float64)
            dist_from_side = coord_along if first else (axis_len - 1 - coord_along)
            proximity = 1.0 + self._stump_surface_proximity_bonus * np.clip(
                1.0 - dist_from_side / depth_ref, 0.0, 1.0
            )
            w *= proximity
            w = np.maximum(w, 1e-9)
            probs = w / w.sum()
            chosen = indices[np.random.choice(len(indices), p=probs)]
            pos_3d = list(self._vox_2_unit_pos(tuple(chosen)))
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
