import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from typing import Tuple

from vessel_graph_generation.forest import Forest
from vessel_graph_generation.simulation_space import SimulationSpace
from vessel_graph_generation.utilities import eukledian_dist, norm_vector, normalize_vector, get_angle_between_vectors, get_angle_between_two_vectors
from vessel_graph_generation.arterial_tree import Node
from vessel_graph_generation.element_mesh import CoordKdTree, NodeKdTree, SpacePartitioner
from tqdm import tqdm
import time

class Greenhouse():

    def __init__(self, config: dict):
        self.config = config
        self.modes: list[dict] = config['modes']
        
        self.sigma_t: float = 1
        self.param_scale: float = config['param_scale']
        self.d: float = config['d'] / self.param_scale
        self.r: float = config['r'] / self.param_scale
        self.FAZ_radius = np.random.normal(config['FAZ_radius_bound'][0] / self.param_scale, config['FAZ_radius_bound'][1] / self.param_scale)
        self.rotation_radius: float = config['rotation_radius'] / self.param_scale
        self.FAZ_center: Tuple[float, float] = config['FAZ_center']
        self.nerve_center = np.array(config["nerve_center"]) / self.param_scale
        self.nerve_radius = np.array(config["nerve_radius"]) / self.param_scale
        self.simspace = SimulationSpace(config["SimulationSpace"], self.FAZ_center, self.FAZ_radius, nerve_center=self.nerve_center, nerve_radius=self.nerve_radius)

        self.init_params_from_config(self.modes[0])

    def init_params_from_config(self, config: dict):
        self.I: int = config['I']
        self.N: int = config['N']
        self.eps_n: float = config['eps_n']
        self.eps_s: float = config['eps_s']
        self.eps_k: float = config['eps_k']
        self.delta_art: float = config['delta_art']
        self.delta_ven: float = config['delta_ven']
        self.gamma_art: float = config['gamma_art']
        self.gamma_ven: float = config['gamma_ven']
        self.phi: float = config['phi']
        self.omega: float = config['omega']
        self.kappa: float = config['kappa']
        self.delta_sigma: float = config['delta_sigma']
        self.sigma_t: float = 1
        
        self.orig_scale = [param / self.param_scale for param in [self.eps_k, self.eps_n, self.eps_s, self.delta_art, self.delta_ven]]
        self.orig_scale.append(self.d)

    def set_forests(self, arterialForest: Forest, venousForest: Forest = None):
        self.arterial_forest = arterialForest
        self.venous_forest = venousForest

    def develop_forest(self):
        """
        Main loop. Generates the arterial (and venous) blood vessel forest
        """
        self.art_node_mesh = NodeKdTree()
        self.art_node_mesh.extend(list(self.arterial_forest.get_nodes()))
        self.active_art_node_mesh = NodeKdTree()
        self.active_art_node_mesh.extend(list(self.arterial_forest.get_nodes()))
        if self.venous_forest is not None:
            self.ven_node_mesh = NodeKdTree()
            self.ven_node_mesh.extend(list(self.venous_forest.get_nodes()))
            self.active_ven_node_mesh = NodeKdTree()
            self.active_ven_node_mesh.extend(list(self.venous_forest.get_nodes()))

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
        for mode in self.modes:
            if mode["name"] != self.modes[0]["name"]:
                self.init_params_from_config(mode)
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
                to_add = set()
                for node in new_nodes:
                    for oxy in self.oxy_mesh.find_elements_in_distance(node.position, self.eps_k):
                        # Oxygen sink is satisfied. Transform to CO2 source
                        to_remove.add(oxy)
                        if self.venous_forest is not None:
                            closest = self.ven_node_mesh.find_nearest_element(oxy, self.eps_k)
                            if closest is None:
                                # self.co2_mesh.add(oxy)
                                to_add.add(oxy)
                        # self.oxy_mesh.delete(oxy)
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
                avg_attraction_vector = sum([norm_vector(att-node.position) for att in atts])

                angles = angles_i[valid_inds]
                # IFF the standard deviation of the angles formed by the attraction vectors is larger than a predefined threshold φ
                if np.std(angles) > self.phi and (self.FAZ_radius==0 or ((dist_to_center / (2*self.FAZ_radius))**5 > random.uniform(0,1) and get_angle_between_two_vectors(vector_to_center, avg_attraction_vector[:2])>90)):
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

                    new_nodes.append(node.tree.add_node(p_new_1, r_1, node, self.kappa))
                    new_nodes.append(node.tree.add_node(p_new_2, r_2, node, self.kappa))
                    # Update raddi of all parent edges up to root with Murray
                    node.optimize_edge_radius_to_root()
                    node_mesh.delete(node)
                else:
                    # Elongation
                    g = self.omega*norm_vector(v) + (1-self.omega)*norm_vector(sum([norm_vector(att-node.position) for att in atts]))
                    
                    if self.rotation_radius>0 and t>15:
                        # Apply quadratic increasing weight to grow in circle near the FAZ
                        g = norm_vector(g)
                        center_vector = norm_vector(np.array(self.FAZ_center)-node.position[:2])
                        dist_new_pos_to_center = np.linalg.norm(np.array(self.FAZ_center)-(node.position + self.d * g)[:2])
                        weight = max(0.01 if not first_mode else 0,self.rotation_radius-dist_new_pos_to_center)
                        weight = sqrt(weight)
                        ort_vector = np.array([-center_vector[1],center_vector[0],0])
                        if get_angle_between_two_vectors(g[:2],ort_vector[:2])>90:
                            ort_vector = -1*ort_vector
                        out_vector = np.array([-center_vector[0],-center_vector[1],0])
                        g = (1-weight)*g + 0.7*weight*ort_vector + 0.3*weight*out_vector

                    p_k = np.real(node.position + self.d * norm_vector(g))
                    new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa))
            elif node.is_inter_node:
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

                avg_attraction_vector = sum([norm_vector(att-node.position) for att in atts])
                # Rodrigues' rotation formula
                # If v is a vector in ℝ3 and k is a unit vector describing an axis of rotation about which 
                # v rotates by an angle θ according to the right hand rule, the Rodrigues formula for the rotated vector vrot is 
                # v_rot = v*cos(θ) + (k×v)sin(θ) + k(k·v)(1-cos(θ))
                # Rotation axis is cross product of child-vector and average attraction vector
                distal_vector = norm_vector(node.get_distal_segment())
                cross = np.cross(distal_vector, avg_attraction_vector)
                if all(cross==0) or ((dist_to_center / (2*self.FAZ_radius))**5 <= random.uniform(0,1) and get_angle_between_two_vectors(vector_to_center, avg_attraction_vector[:2])<=90):
                    continue
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
                new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa))
                # Update raddi of all parent edges up to root with Murray
                node.optimize_edge_radius_to_root()
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
            if all([eukledian_dist(candidate_sink, node.position) > self._calculate_oxygen_distance(node.radius) for node in self.art_node_mesh.find_elements_in_distance(candidate_sink, eps_n)]) \
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
        fig = plt.figure(figsize=((12, 12)))
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

        co2s = np.array(self.co2_mesh.get_all_elements())
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

        plt.plot(self.art_nodes_per_step)
        plt.plot(self.oxys_per_step)
        if self.venous_forest is not None:
            plt.plot(self.ven_nodes_per_step)
            plt.plot(self.co2_per_step)
            plt.legend(['Arterial Nodes', 'Oxygen Sinks', 'Venous Nodes', 'CO₂ Sources'])
        else:
            plt.legend(['Nodes', 'Oxygen Sinks'])
        plt.title('Growth Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Amount')
        plt.savefig(f'{out_dir}/growth_over_time.png', bbox_inches='tight')
        plt.close()
