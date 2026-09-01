import time
import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vessel_graph_generation.arterial_tree import Node
from vessel_graph_generation.element_mesh import (
    CoordKdTree,
    NodeKdTree,
    SpacePartitioner,
)
from vessel_graph_generation.faz_extraction import (
    extract_faz_center_radius_unit,
    load_raw_geometry,
)
from vessel_graph_generation.forest import Forest
from vessel_graph_generation.simulation_space import SimulationSpace
from vessel_graph_generation.utilities import (
    eukledian_dist,
    get_angle_between_two_vectors,
    get_angle_between_vectors,
    norm_vector,
    normalize_vector,
)
import seaborn as sns

sns.set_theme()


class Greenhouse():

    def __init__(self, config: dict):
        self.config = config
        self.modes: list[dict] = config['modes']
        
        self.sigma_t: float = 1
        self.param_scale: float = config['param_scale']
        self.d: float = config['d'] / self.param_scale
        self.r: float = config['r'] / self.param_scale
        # rotation_radius is set after FAZ_radius is finalized (see rotation_radius_from_faz below).
        # Raw FAZ radius from geometry (unit space), before FAZ_radius_calibration; None if not from geometry.
        self.FAZ_radius_geometry: float | None = None

        used_geometry_faz_radius: bool = False
        if config.get('FAZ_from_geometry', False):
            sim_cfg = config['SimulationSpace']
            geom_path = sim_cfg.get('oxygen_sample_geometry_path')
            if not geom_path:
                raise ValueError("FAZ_from_geometry requires SimulationSpace.oxygen_sample_geometry_path")
            proj_axis = int(config.get('FAZ_geometry_proj_axis', 2))
            raw_geom = load_raw_geometry(geom_path)
            geom_size = int(max(raw_geom.shape))
            faz_center_arr, faz_r_unit = extract_faz_center_radius_unit(
                raw_geom, proj_axis=proj_axis, geometry_size=geom_size
            )
            fallback_ok = config.get('FAZ_geometry_use_config_fallback', True)
            if (
                not np.all(np.isfinite(faz_center_arr))
                or not np.isfinite(faz_r_unit)
                or faz_r_unit <= 0
            ):
                msg = (
                    "Could not derive FAZ from geometry (empty or invalid avascular region); "
                )
                if fallback_ok:
                    warnings.warn(msg + "using FAZ_center and FAZ_radius_bound from config.")
                    self.FAZ_radius = float(
                        np.random.normal(
                            config['FAZ_radius_bound'][0] / self.param_scale,
                            config['FAZ_radius_bound'][1] / self.param_scale,
                        )
                    )
                    self.FAZ_center = tuple(float(x) for x in config['FAZ_center'])
                else:
                    raise RuntimeError(msg + "Set FAZ_geometry_use_config_fallback: true or fix the geometry.")
            else:
                self.FAZ_center = (float(faz_center_arr[0]), float(faz_center_arr[1]))
                self.FAZ_radius_geometry = float(faz_r_unit)
                self.FAZ_radius = float(faz_r_unit)
                used_geometry_faz_radius = True
        else:
            self.FAZ_radius = float(
                np.random.normal(
                    config['FAZ_radius_bound'][0] / self.param_scale,
                    config['FAZ_radius_bound'][1] / self.param_scale,
                )
            )
            self.FAZ_center = tuple(float(x) for x in config['FAZ_center'])

        cal = config.get("FAZ_radius_calibration") or {}
        if cal.get("enabled", False):
            apply_only = cal.get("apply_only_after_geometry_extraction", True)
            if used_geometry_faz_radius or not apply_only:
                a = float(cal["a"])
                b = float(cal["b"])
                r_given = float(self.FAZ_radius)
                r_calibrated = a * r_given + b
                if r_calibrated < 0:
                    warnings.warn(
                        f"FAZ_radius_calibration: pre-calibration FAZ_radius={r_given}, "
                        f"calibrated value would be {r_calibrated}; "
                        f"clamping FAZ_radius to 0.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.FAZ_radius = 1e-5
                else:
                    self.FAZ_radius = r_calibrated

        # rotation_radius is in unit space. Either fixed (config['rotation_radius'] mm / param_scale),
        # or linearly derived from the (already calibrated) FAZ_radius via
        #   rotation_radius = scale * FAZ_radius + offset_mm / param_scale
        rr_from_faz = config.get("rotation_radius_from_faz") or {}
        if rr_from_faz.get("enabled", False):
            scale = float(rr_from_faz.get("scale", 1.0))
            offset_mm = float(rr_from_faz.get("offset_mm", 0.0))
            self.rotation_radius = scale * self.FAZ_radius + offset_mm / self.param_scale
            if self.rotation_radius < 0:
                warnings.warn(
                    f"rotation_radius_from_faz: derived rotation_radius={self.rotation_radius} < 0 "
                    f"(scale={scale}, offset_mm={offset_mm}, FAZ_radius={self.FAZ_radius}); "
                    "clamping to 0.",
                    UserWarning,
                    stacklevel=2,
                )
                self.rotation_radius = 0.0
        else:
            self.rotation_radius = float(config['rotation_radius']) / self.param_scale

        # Dimensionless scale on rotation blend weight (after sqrt in grow_vessels).
        self.rotation_strength: float = float(config.get('rotation_strength', 1.0))
        self.prune_leaf_neighbors_of_degree3: bool = bool(config.get('prune_leaf_neighbors_of_degree3', False))

        self.nerve_center = np.array(config["nerve_center"]) / self.param_scale
        self.nerve_radius = np.array(config["nerve_radius"]) / self.param_scale
        self.simspace = SimulationSpace(config["SimulationSpace"], self.FAZ_center, self.FAZ_radius, nerve_center=self.nerve_center, nerve_radius=self.nerve_radius)

        self.init_params_from_config(self.modes[0])

    def init_params_from_config(self, config: dict, update_scale_params: bool = True):
        self.I: int = config['I']
        self.N: int = config['N']
        self.gamma_art: float = config['gamma_art']
        self.gamma_ven: float = config['gamma_ven']
        self.phi: float = config['phi']
        self.omega: float = config['omega']
        self.allow_bifurcation: bool = config.get('allow_bifurcation', True)
        self.min_bifurcation_hops: int = int(config.get('min_bifurcation_hops', 0))
        self.kappa: float = config['kappa']
        self.delta_sigma: float = config['delta_sigma']
        
        if update_scale_params:
            self.sigma_t: float = 1
            self.eps_n: float = config['eps_n']
            self.eps_s: float = config['eps_s']
            self.eps_k: float = config['eps_k']
            self.delta_art: float = config['delta_art']
            self.delta_ven: float = config['delta_ven']
            self.orig_scale = [param / self.param_scale for param in [self.eps_k, self.eps_n, self.eps_s, self.delta_art, self.delta_ven]]
            self.orig_scale.append(self.d)

    def set_forests(self, arterialForest: Forest, venousForest: Forest = None):
        self.arterial_forest = arterialForest
        self.venous_forest = venousForest

    def _prune_active_inter_nodes_near_degree3(self, branch_node: Node, active_node_mesh: SpacePartitioner[Node]):
        # Once a node became degree-3, suppress nearby degree-2 nodes as sprouting candidates.
        to_remove = branch_node.get_blocked_inter_nodes_after_bifurcation()
        if to_remove:
            active_node_mesh.delete_all(to_remove)

    def _prune_leaf_neighbors_of_degree3_after_growth(self):
        """
        Optional post-processing: remove leaf nodes whose direct parent has graph degree 3.
        """
        def _collect_prunable_leaves(forest: Forest) -> list[Node]:
            leaves_to_remove: list[Node] = []
            for tree in forest.get_trees():
                for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    if node.is_leaf and node.parent is not None and node.parent.graph_degree() == 3:
                        leaves_to_remove.append(node)
            return leaves_to_remove

        art_to_remove = _collect_prunable_leaves(self.arterial_forest)
        if art_to_remove:
            self.art_node_mesh.delete_all(art_to_remove)
            self.active_art_node_mesh.delete_all(art_to_remove)
            for node in art_to_remove:
                node.parent = None

        if self.venous_forest is not None:
            ven_to_remove = _collect_prunable_leaves(self.venous_forest)
            if ven_to_remove:
                self.ven_node_mesh.delete_all(ven_to_remove)
                self.active_ven_node_mesh.delete_all(ven_to_remove)
                for node in ven_to_remove:
                    node.parent = None

    def develop_forest(self):
        """
        Main loop. Generates the arterial (and venous) blood vessel forest
        """
        self.art_node_mesh = NodeKdTree()
        self.art_node_mesh.extend(list(self.arterial_forest.get_nodes()))
        self.active_art_node_mesh = NodeKdTree()
        self.active_art_node_mesh.extend(list(self.arterial_forest.get_nodes()))
        for node in self.art_node_mesh.get_all_elements():
            if not hasattr(node, "min_bifurcation_hops"):
                node.min_bifurcation_hops = self.min_bifurcation_hops
        if self.venous_forest is not None:
            self.ven_node_mesh = NodeKdTree()
            self.ven_node_mesh.extend(list(self.venous_forest.get_nodes()))
            self.active_ven_node_mesh = NodeKdTree()
            self.active_ven_node_mesh.extend(list(self.venous_forest.get_nodes()))
            for node in self.ven_node_mesh.get_all_elements():
                if not hasattr(node, "min_bifurcation_hops"):
                    node.min_bifurcation_hops = self.min_bifurcation_hops

        # Stats
        self.art_nodes_per_step = [0]
        self.oxys_per_step = [0]
        self.ven_nodes_per_step = [0]
        self.co2_per_step = [0]
        self.time_per_step = []

        self.oxy_mesh = CoordKdTree()
        if self.venous_forest is not None:
            self.co2_mesh = CoordKdTree()
        t = 0
        # mbar = tqdm(self.modes)
        for mode_idx, mode in enumerate(self.modes):
            self.init_params_from_config(mode, update_scale_params=(mode["name"] != self.modes[max(0,mode_idx-1)]["name"]))
            if mode_idx > 0 and self.simspace.multi_zone:
                # Transition to DVC mode if multi-zone geometry is used
                self._transition_to_dvc_mode(mode["name"])
            if self.I<=0:
                continue

            # pbar = tqdm(range(t,t+self.I), desc=f'[{mode}] Art: {self.art_nodes_per_step[-1]},Oxy: {self.oxys_per_step[-1]}, Ven: {self.ven_nodes_per_step[-1]}, Co2: {self.co2_per_step[-1]}')
            for t in range(t,t+self.I):
                s = time.time()
                # 1. Sample oxygen sinks
                self.sample_oxygen_sinks(int(self.N), max(self.eps_n, self.eps_k), self.eps_s, t=t)
                # 2. Arterial vessel growth
                new_nodes = self.grow_vessels(self.active_art_node_mesh, self.oxy_mesh, self.gamma_art, self.delta_art, first_mode=mode == self.modes[0], t=t)
                self.art_node_mesh.extend(new_nodes)
                self.active_art_node_mesh.extend(new_nodes)
                # 3. All oxygen sinks within distance d_suff of arterial nodes are converted to carbon-dioxid sources
                to_remove = set()
                for node in new_nodes:
                    to_remove.update(self.oxy_mesh.find_elements_in_distance(node.position, self.eps_k))
                
                # Check which satisfied oxygen sinks should become CO2 sources
                to_add = set()
                if self.venous_forest is not None:
                    for oxy in to_remove:
                        closest = self.ven_node_mesh.find_nearest_element(oxy, self.eps_k)
                        if closest is None:
                            to_add.add(oxy)
                
                    self.co2_mesh.extend(to_add)
                self.oxy_mesh.delete_all(to_remove)

                if self.venous_forest is not None:
                    # 4. Venous vessel growth with δ = 0.07, γ = 100°
                    new_nodes = self.grow_vessels(self.active_ven_node_mesh, self.co2_mesh, self.gamma_ven, self.delta_ven, first_mode=mode == self.modes[0], t=t)
                    self.ven_node_mesh.extend(new_nodes)
                    self.active_ven_node_mesh.extend(new_nodes)
                    # 5. All co2 sinks within distance d_suff of venous nodes are removed
                    to_remove = set()
                    for node in new_nodes:
                        to_remove.update(self.co2_mesh.find_elements_in_distance(node.position, self.eps_k))
                    self.co2_mesh.delete_all(to_remove)
                # 6. Scaling
                self.simulation_space_expansion()

                # Update stats
                self.time_per_step.append(time.time()-s)
                self.art_nodes_per_step.append(len(self.art_node_mesh.get_all_elements()))
                self.oxys_per_step.append(len(self.oxy_mesh.get_all_elements()))

                if self.venous_forest is not None:
                    self.ven_nodes_per_step.append(len(self.ven_node_mesh.get_all_elements()))
                    self.co2_per_step.append(len(self.co2_mesh.get_all_elements()))
                    # pbar.set_description(f'[{mode}] Art: {self.art_nodes_per_step[-1]}, Oxy: {self.oxys_per_step[-1]}, Ven: {self.ven_nodes_per_step[-1]}, CO2: {self.co2_per_step[-1]}')
                # else:
                #     pbar.set_description(f'[{mode}] Art: {self.art_nodes_per_step[-1]}, Oxy: {self.oxys_per_step[-1]}')

        if self.prune_leaf_neighbors_of_degree3:
            self._prune_leaf_neighbors_of_degree3_after_growth()
            # Record post-processing as one extra stats step so node removal is visible.
            self.time_per_step.append(0.0)
            self.art_nodes_per_step.append(len(self.art_node_mesh.get_all_elements()))
            self.oxys_per_step.append(len(self.oxy_mesh.get_all_elements()))
            if self.venous_forest is not None:
                self.ven_nodes_per_step.append(len(self.ven_node_mesh.get_all_elements()))
                self.co2_per_step.append(len(self.co2_mesh.get_all_elements()))
    
    def _transition_to_dvc_mode(self, mode_name: dict):
        """
        Transition from SVC to DVC growth phase (only for multi-zone geometries).
        
        This method:
        1. Switches simulation space to DVC zone
        2. Shadows SVC nodes with degree < 3 into the DVC layer
        3. Moves all oxygen sinks from SVC to DVC
        
        Parameters:
        -----------
        - mode_name: Name of the new growth mode (should correspond to DVC)
        """
        # Step 1: Switch simulation space to DVC zone
        current_zone_value = self.simspace.zones[self.simspace.current_zone]["value"]
        
        self.simspace.set_active_zone(mode_name)

        new_zone_value = self.simspace.zones[self.simspace.current_zone]["value"]
        if current_zone_value == new_zone_value:
            return
        
        # Step 2: Shadow eligible SVC nodes (degree < 3, not root, not already shadowed)
        shadow_nodes = []
        for node in list(self.art_node_mesh.get_all_elements()):
            if (node.get_degree() < 3 and 
                not node.is_root and 
                not node.is_shadowed and
                node.shadow_pair is None):  # Don't shadow nodes that already have a shadow
                
                # Random depth in DVC zone
                dvc_z_min, dvc_z_max = self.simspace.get_zone_height_range(mode_name, node.position[0], node.position[1])
                if dvc_z_min == -1 or dvc_z_max == -1 or np.random.uniform(0,1) > np.exp(-0.1 * self.simspace.geometry_size * (dvc_z_min-node.position[2])):
                    continue  # Skip if no valid DVC range at this (x,y)
                shadow_depth = np.random.triangular(left=dvc_z_min, mode=dvc_z_min, right=dvc_z_max) if dvc_z_min != dvc_z_max else dvc_z_min
                shadow_node = node.create_shadow(shadow_depth)
                shadow_nodes.append(shadow_node)
        
        # Step 3: Move oxygen sinks from SVC to DVC
        self._relocate_oxygen_sinks_to_dvc(mode_name)
        
        # Add shadow nodes to mesh for attraction computation
        self.art_node_mesh.extend(shadow_nodes)
        self.active_art_node_mesh.extend(shadow_nodes)
    
    def _relocate_oxygen_sinks_to_dvc(self, mode_name: str):
        """
        Move all oxygen sinks from SVC layer to random DVC depths.
        
        Parameters:
        -----------
        - z_min: Minimum z-coordinate in DVC layer
        - z_max: Maximum z-coordinate in DVC layer
        """
        all_oxys = list(self.oxy_mesh.get_all_elements())
        self.oxy_mesh.delete_all(all_oxys)
        
        relocated = []
        for oxy in all_oxys:
            # Keep x, y but randomize z within DVC range
            z_min, z_max = self.simspace.get_zone_height_range(mode_name, oxy[0], oxy[1])
            if z_min == -1 or z_max == -1:
                continue  # Skip if no valid DVC range at this (x,y)
            new_oxy = (oxy[0], oxy[1], np.random.uniform(z_min, z_max))
            relocated.append(new_oxy)
        
        self.oxy_mesh.extend(relocated)

    def simulation_space_expansion(self):
        """
        Scales all distance related parameters to simulate the expansion of the simulation space.
        This is motivated by the growth of tissue in real life.
        """
        # scaling factor at time t: σ_t = σ_t−1 +∆σ
        self.sigma_t = self.sigma_t + self.delta_sigma
        self.eps_k, self.eps_n, self.eps_s, self.delta_art, self.delta_ven, self.d = [param / self.sigma_t for param in self.orig_scale]
        self.d = max(self.d, 0.04/self.param_scale)
        
        self.art_node_mesh.reassign(self.delta_art)
        self.active_art_node_mesh.reassign(self.delta_art)
        self.oxy_mesh.reassign(self.delta_art)
        if self.venous_forest is not None:
            self.ven_node_mesh.reassign(self.delta_ven)
            self.active_ven_node_mesh.reassign(self.delta_ven)
            self.co2_mesh.reassign(self.delta_ven)
    
    def grow_vessels(self, node_mesh: SpacePartitioner[Node], att_mesh: SpacePartitioner, gamma: float, delta: float, first_mode=True, t=0) -> list[Node]:
        """
        Performs arterial or venous vessel growth

        Paramters:
        ---------
        - node_mesh: NodeMesh of vessel nodes that are grown
        - att_mesh: CoordsMesh of oxygen or co2 sinks

        Returns:
        --------
        List of all new Nodes that were added in this iteration
        """
        # Nearest Neighbor search
        att_node_assignment: dict[Node, list[tuple[float]]] = self.assign_attraction_points_to_node(node_mesh, att_mesh, delta)
        new_nodes: list[Node] = []
        # Vessel Growth
        for node, atts in att_node_assignment.items():
            
            vector_to_center = np.array(self.FAZ_center)-node.position[:2]
            dist_to_center = np.linalg.norm(vector_to_center)
            if node.is_leaf:
                v = node.get_proximal_segment()
                angles_i = get_angle_between_vectors(v, atts - node.position)
                # Requirements for oxygen sink with position p_0, distance r and angle θ
                #   - r = ||p_o − p_j||<=δ
                #   - θ = cos^−1 e_ij * nrm(p_o − p_j) <= γ/2
                valid_inds = angles_i <= max(gamma/2,0)
                atts = np.array(atts)[valid_inds]
                if len(atts) == 0:
                    continue
                
                if node.is_shadowed:
                    # Materialize before sprouting
                    node.is_shadowed = False

                # Vectorized normalization
                diff_vectors = atts - node.position
                norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
                avg_attraction_vector = np.sum(diff_vectors / norms, axis=0)

                angles = angles_i[valid_inds]
                # IFF the standard deviation of the angles formed by the attraction vectors is larger than a predefined threshold φ
                if self.allow_bifurcation and np.std(angles) > self.phi and (
                    self.FAZ_radius==0 or (
                        (dist_to_center / (2*self.FAZ_radius))**5 > np.random.uniform(0,1) and 
                        get_angle_between_two_vectors(vector_to_center, avg_attraction_vector[:2])>90
                    )):
                    # Bifurcation:
                    
                    # # Radii of two resulting vessels after Murray's law
                    # r_p = node.get_proximal_radius()
                    # r_c = 2 ** (-1 / kappa) * r_p
                    # # Sample first radius from a normal dist
                    # r_1 = norm.rvs(loc=r_c, scale=r_c / 32)
                    # r_2 = (r_p ** kappa - r_1 ** kappa) ** (1 / kappa)
                    # r_1 = np.clip(r_1, self.r_min, r_p)
                    # r_2 = np.clip(r_2, self.r_min, r_p)

                    # Radii with fix terminal length r
                    r_1 = r_2 = self.r
                    # r_1,r_2 = np.random.default_rng().normal((0.3/node.proximal_num_segments + 0.8)*self.r, self.r / 25, 2)
                    r_p = (r_1**self.kappa + r_2**self.kappa)**(1/self.kappa)

                    d1 = d2 = self.d#np.random.default_rng().normal((0.3/node.proximal_num_segments + 0.8)*self.d, self.d / 25, 2)
                    # # Sample lengths of two resulting vessels from a log-normal dist
                    # l_1 = lognorm.rvs(scale=self.mu_b, s=self.sigma_b) * r_1
                    # l_2 = lognorm.rvs(scale=self.mu_b, s=self.sigma_b) * r_2

                    # Its bifurcation angle is dictated by Murphy's law
                    phi_1 = np.degrees(np.arccos((r_p ** 4 + r_1 ** 4 - r_2 ** 4) / (2 * r_p ** 2 * r_1 ** 2)))
                    phi_2 = np.degrees(np.arccos((r_p ** 4 + r_2 ** 4 - r_1 ** 4) / (2 * r_p ** 2 * r_2 ** 2)))

                    # vector of the new nodes is calculated by:
                    # p_new,1 = node.position + (cos(phi_1)*d_parent + sin(phi_1)*d_l) * d
                    # p_new,2 = node.position + (cos(phi_2)*d_parent + sin(phi_2)*d_l) * d

                    # d_parent = normalize_vector(node.get_proximal_segment())

                    c = np.mean(atts, axis=0)

                    d_parent_c = normalize_vector(c-node.position)

                    X = np.array([oxy-c for oxy in atts]).transpose()
                    X_cov = np.cov(X)
                    w, v = np.linalg.eig(X_cov)
                    d_l = v[:, np.argmax(w)]

                    p_new_1 = np.real(node.position + norm_vector(np.cos(np.radians(phi_1)) * d_parent_c + np.sin(np.radians(phi_1))*d_l) * d1)
                    p_new_2 = np.real(node.position + norm_vector(np.cos(np.radians(phi_2)) * d_parent_c - np.sin(np.radians(phi_2))*d_l) * d2)

                    new_nodes.append(node.tree.add_node(p_new_1, r_1, node, self.kappa, self.min_bifurcation_hops))
                    new_nodes.append(node.tree.add_node(p_new_2, r_2, node, self.kappa, self.min_bifurcation_hops))
                    # Update raddi of all parent edges up to root with Murray
                    # if first_mode:
                    node.optimize_edge_radius_to_root()
                    self._prune_active_inter_nodes_near_degree3(node, node_mesh)
                    node_mesh.delete(node)
                else:
                    # Elongation
                    # Vectorized normalization for attraction sum
                    diff_vectors = atts - node.position
                    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
                    attraction_sum = np.sum(diff_vectors / norms, axis=0)
                    g = self.omega*norm_vector(v) + (1-self.omega)*norm_vector(attraction_sum)
                    
                    if self.rotation_radius>0 and t>15:
                        # Apply quadratic increasing weight to grow in circle near the FAZ
                        g = norm_vector(g)
                        center_vector = norm_vector(np.array(self.FAZ_center)-node.position[:2])
                        dist_new_pos_to_center = np.linalg.norm(np.array(self.FAZ_center)-(node.position + self.d * g)[:2])
                        weight = max(0.01 if not first_mode else 0,self.rotation_radius-dist_new_pos_to_center)
                        weight = sqrt(weight)
                        weight *= self.rotation_strength
                        ort_vector = np.array([-center_vector[1],center_vector[0],0])
                        if get_angle_between_two_vectors(g[:2],ort_vector[:2])>90:
                            ort_vector = -1*ort_vector
                        out_vector = np.array([-center_vector[0],-center_vector[1],0])
                        g = (1-weight)*g + 0.7*weight*ort_vector + 0.3*weight*out_vector

                    p_k = np.real(node.position + self.d * norm_vector(g))
                    new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa, self.min_bifurcation_hops))
            elif self.allow_bifurcation and node.is_inter_node:
                if not node.can_inter_node_sprout():
                    continue
                # Calculate optimal radius and angle with Murray
                r_1 = node.get_distal_radius()
                r_2 = self.r # np.random.default_rng().normal((0.3/node.proximal_num_segments + 0.8)*self.r, self.r / 25)

                r_p = (r_1**self.kappa + r_2**self.kappa)**(1/self.kappa)
                phi_1 = np.degrees(np.arccos((r_p ** 4 + r_1 ** 4 - r_2 ** 4) / (2 * r_p ** 2 * r_1 ** 2)))
                phi_2 = np.degrees(np.arccos((r_p ** 4 + r_2 ** 4 - r_1 ** 4) / (2 * r_p ** 2 * r_2 ** 2)))
                
                # Reqirements for oxygen sinks:
                #   - α−γ/2 <= θ <= α+γ/2
                #   - r ≤ δ
                angles_distal = get_angle_between_vectors(node.get_distal_segment(0), atts - node.position)
                angles_proximal = get_angle_between_vectors(node.get_proximal_segment(), atts - node.position)
                atts = np.array(atts)[
                    (phi_1 + phi_2 - gamma/2 <= angles_distal) &
                    (angles_distal <= (phi_1 + phi_2 + gamma/2)) &
                    (angles_proximal<= phi_2 + gamma/2)
                ]
                if len(atts) == 0:
                    continue

                # Vectorized normalization
                diff_vectors = atts - node.position
                norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
                avg_attraction_vector = np.sum(diff_vectors / norms, axis=0)
                # Rodrigues' rotation formula
                # If v is a vector in ℝ3 and k is a unit vector describing an axis of rotation about which 
                # v rotates by an angle θ according to the right hand rule, the Rodrigues formula for the rotated vector vrot is 
                # v_rot = v*cos(θ) + (k×v)sin(θ) + k(k·v)(1-cos(θ))
                # Rotation axis is cross product of child-vector and average attraction vector
                distal_vector = norm_vector(node.get_distal_segment())
                cross = np.cross(distal_vector, avg_attraction_vector)
                if all(cross==0) or ((dist_to_center / (2*self.FAZ_radius))**5 <= np.random.uniform(0,1) and get_angle_between_two_vectors(vector_to_center, avg_attraction_vector[:2])<=90):
                    continue
                
                if node.is_shadowed:
                    # Materialize before sprouting
                    node.is_shadowed = False

                rot_axis = norm_vector(cross)
                theta = phi_2 #get_angle_between_vectors(distal_vector, avg_attraction_vector)
                # Calculate hypothetical optimal branch closest to the average attraction vector
                v = distal_vector*np.cos(np.radians(theta)) + np.cross(rot_axis, distal_vector)*np.sin(np.radians(theta)) \
                        + rot_axis*np.dot(rot_axis, distal_vector)*(1-np.cos(np.radians(theta)))
                # if get_angle_between_vectors(v, node.get_proximal_segment()) > 180:
                #     v = -v
                g = self.omega*norm_vector(v) + (1-self.omega)*norm_vector(avg_attraction_vector)
                
                d = self.d#np.random.default_rng().normal((0.3/node.proximal_num_segments + 0.8)*self.d, self.d / 25)

                p_k = np.real(node.position + d * norm_vector(g))
                new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa, self.min_bifurcation_hops))
                # Update raddi of all parent edges up to root with Murray
                # if first_mode:
                node.optimize_edge_radius_to_root()
                self._prune_active_inter_nodes_near_degree3(node, node_mesh)
                node_mesh.delete(node)
        return new_nodes

    def _calculate_oxygen_distance(self, r): 
        """
        Models the oxygen concentration heuristic by Schneider et al., 2012 (https://doi.org/10.1016/j.media.2012.04.009)
        """
        c_oxygen = 203.9e-3 # oxygen concentration inside vessel lumen in m^3 per m^3
        kappa = 0.02 * c_oxygen # peak perfusion level in simulation space
        r0 = 3.5e-3 # vessel radius perfusing maximum amount of oxygen in mm
        c1 = kappa * (r*self.param_scale/r0)*np.exp(1-(r*self.param_scale/r0))
        return c1 * 6 / self.param_scale

    def sample_oxygen_sinks(self, N=1000, eps_n=0.04, eps_s=0.3, t=0) -> list[tuple[float]]:
        """
        Sample Oyigen sinks from hypoxic tissue.

        Parameters:
        -----------
        - N: Number of tries
        - eps_n: threshold to arterial node within which samples are discarded
        - eps_s: threshold to other oxigen sinks within which samples are discarded
        - radius: Radius in the center of the simulation space where no oxygen sinks are sampled from

        Returns:
        -------
        List of 3D coordinates of valid oxygen sinks
        """
        to_add = list()
        candidate_sinks = self.simspace.get_candidate_sinks(N)
        for candidate_sink in candidate_sinks:
            if all([eukledian_dist(candidate_sink, node.position) > self._calculate_oxygen_distance(node.radius) for node in self.art_node_mesh.find_elements_in_distance(candidate_sink, eps_n) if not node.is_shadowed]) \
                and self.oxy_mesh.find_nearest_element(candidate_sink, eps_s) is None \
                and (not to_add or all(np.linalg.norm(np.array(candidate_sink)-np.array(to_add), axis=1) > eps_s)):
                to_add.append(tuple(candidate_sink))
        self.oxy_mesh.extend(to_add)

    def assign_attraction_points_to_node(self, node_mesh: SpacePartitioner[Node], attraction_point_mesh: SpacePartitioner, delta: float) -> dict[Node, list[tuple[float]]]:
        """
        Performs nearest neighbor search to assign each attraction point to its closest vessel node
        
        Paramters:
        ---------
        node_mesh: NodeMesh of nodes of interest
        attraction_point_mesh: CoordsMesh of oxygen or co2 sinks of interest

        Returns:
        -------
        Node to attraction points dictionary where each attraction point is assigned to its closest node.
        List where attractions points at index i are closest to node i in forest
        """
        assignment = dict()
        for attraction_point in attraction_point_mesh.get_all_elements():
            closest = node_mesh.find_nearest_element(attraction_point, max_dist=delta)
            if closest is None:
                continue
            if closest in assignment:
                assignment[closest].append(attraction_point)
            else:
                assignment[closest] = [attraction_point]
        return assignment

    def save_forest_fig(self, output_path='output.png'):
        plt.figure(figsize=((12, 12)))
        ax = plt.axes(projection='3d')
        forests = [self.arterial_forest]
        if self.venous_forest is not None:
            forests.append(self.venous_forest)
        for forest in forests:
            for tree in  tqdm(forest.get_trees(), desc="Preparing figure"):
                size_x = tree.size_x
                size_y = tree.size_y
                size_z = tree.size_z

                edges = np.array([np.concatenate([node.parent.position, node.position]) for node in tree.get_tree_iterator(exclude_root=True)])
                radii = np.array([node.radius for node in tree.get_tree_iterator(exclude_root=True)])
                radii /= radii.max()

                linewidth = 8

                for edge, radius in zip(edges, radii):
                    plt.plot([edge[0], edge[3]], [edge[1], edge[4]], [edge[2], edge[5]], c=plt.cm.jet(radius), linewidth=linewidth * radius, axes=ax)

                ax.set_xlim(0, size_x)
                ax.set_ylim(0, size_y)
                ax.set_zlim(0, size_z)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                ax.view_init(elev=90.0, azim=0.0)
                ax.set_box_aspect((size_x, size_y, size_z))
        plt.savefig(output_path, bbox_inches="tight")
            
    def save_stats(self, out_dir: str):
        plt.figure(figsize=(6,6))
        oxys = np.array(self.oxy_mesh.get_all_elements())
        if len(oxys)>0:
            plt.plot(oxys[:,1], 1-oxys[:,0], 'r.')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title('Final Oxygen Sink Distribution')
        plt.savefig(f'{out_dir}/oxy_distribution.png', bbox_inches='tight')
        plt.cla()

        if self.venous_forest is not None:
            co2s = np.array(self.co2_mesh.get_all_elements())
        else:
            co2s = np.array([])
        if len(co2s)>0:
            plt.plot(co2s[:,1], 1-co2s[:,0], 'b.')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title('Final CO₂ Sink Distribution')
        plt.savefig(f'{out_dir}/co2_distribution.png', bbox_inches='tight')
        plt.cla()

        plt.plot(self.time_per_step)
        total = time.strftime('%H:%M:%S', time.gmtime(sum(self.time_per_step)))
        plt.title(f'Runtime Per Iteration (Total={total})')
        plt.xlabel("Iterations")
        plt.ylabel("Seconds")
        plt.savefig(f'{out_dir}/time_per_step.png', bbox_inches='tight')
        plt.cla()

        (art_line,) = plt.plot(self.art_nodes_per_step)
        (oxy_line,) = plt.plot(self.oxys_per_step)
        final_art_nodes = self.art_nodes_per_step[-1]
        last_t = len(self.art_nodes_per_step) - 1
        plt.plot(
            last_t,
            final_art_nodes,
            "o",
            color=art_line.get_color(),
            zorder=5,
        )
        if self.venous_forest is not None:
            (ven_line,) = plt.plot(self.ven_nodes_per_step)
            (co2_line,) = plt.plot(self.co2_per_step)
            plt.legend(
                (art_line, oxy_line, ven_line, co2_line),
                (
                    f"Arterial Nodes (final: {final_art_nodes})",
                    "Oxygen Sinks",
                    "Venous Nodes",
                    "CO₂ Sources",
                ),
            )
        else:
            plt.legend(
                (art_line, oxy_line),
                (f"Nodes (final: {final_art_nodes})", "Oxygen Sinks"),
            )
        plt.title('Growth Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Amount')
        plt.savefig(f'{out_dir}/growth_over_time.png', bbox_inches='tight')
        plt.close()
