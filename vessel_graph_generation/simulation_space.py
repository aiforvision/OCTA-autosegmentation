import logging
import numpy as np

class SimulationSpace:

    def __init__(self, config):
        if config['oxygen_sample_geometry_path'] != '':
            self.geometry: np.ndarray = np.load(config['oxygen_sample_geometry_path'])
            if len(self.geometry.shape) == 2:
                self.geometry = np.tile(self.geometry[:,:,np.newaxis], config['no_voxel_z'])
        else:
            shape = config["no_voxel_x"], config["no_voxel_y"], config["no_voxel_z"]
            self.geometry: np.ndarray = np.ones(shape)
            x = np.arange(0, shape[0])
            y = np.arange(0, shape[1])
            cx = 99.5
            cy = 99.5
            r = 30
            mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
            self.geometry[mask,:]=0

        if config['tree_root_indices_path'] != '':
            self.valid_start_voxels: list[tuple[int]] = list(zip(*np.load(config['tree_root_indices_path'])))

        self.size_x, self.size_y, self.size_z = self.geometry.shape
        # self.valid_voxels = np.random.uniform(0,1, self.geometry.shape)
        self.valid_voxels = np.array(list(zip(*np.where(self.geometry>0))))

        logging.info('Initialized simulation space of size {} x {} x {}.'.format(self.size_x, self.size_y, self.size_z,))
