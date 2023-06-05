from typing import Literal
import csv
from random import random
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
from numpy import floor

import math
import itertools
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt, collections, cm

from vessel_graph_generation.forest import Forest

def rasterize_forest(forest: dict,
                     image_scale_factor: np.ndarray,
                     radius_list:list=None, 
                     min_radius=0,
                     max_radius=1,
                     max_dropout_prob=0,
                     blackdict: dict[str, bool]=None,
                     colorize=False,
                     continous=True):
    # initialize canvas with defined image dimensions
    if not radius_list:
        radius_list=[]
    size_x = 1
    image_dim = tuple([math.ceil(image_scale_factor * d) for d in [76,76]])
    no_pixels_x, no_pixels_y = image_dim
    voxel_size_x = size_x / (no_pixels_x)
    dpi = 100
    x_inch = no_pixels_x / dpi
    y_inch = no_pixels_y / dpi
    figure = plt.figure(figsize=(x_inch,y_inch))
    figure.patch.set_facecolor('black')
    ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    ax.invert_yaxis()
    edges = []
    radii = []
    if blackdict is None:
        blackdict = dict()
        p = random()**10 * max_dropout_prob
    else:
        p=0
    for edge in forest:
        radius = float(edge["radius"])
        if radius<min_radius or radius>max_radius:
            continue
        radius *= 1.3
        current_node = edge["node1"]
        proximal_node = edge["node2"]

        if isinstance(current_node, np.ndarray) or isinstance(current_node, list):
            current_node = tuple(current_node)
            proximal_node = tuple(proximal_node)
        elif isinstance(current_node, str):
            # Legacy
            current_node = tuple([float(coord) for coord in current_node[1:-1].split(" ") if len(coord)>0])
            proximal_node = tuple([float(coord) for coord in proximal_node[1:-1].split(" ") if len(coord)>0])

        if proximal_node in blackdict or random()<p:
            blackdict[current_node] = True
            continue

        radius_list.append(radius)
        thickness = radius * no_pixels_x
        edges.append([(current_node[1],current_node[0]),(proximal_node[1],proximal_node[0])])
        radii.append(thickness)
    if colorize:
        colors=np.copy(np.array(radii))
        colors = colors/no_pixels_x/1.3*3
        if continous:
            colors=np.minimum(colors/0.03,1)
        else:
            c_new = np.zeros_like(colors)
            c_new[colors<=0.01]=0.1
            c_new[(colors>0.01) & (colors<=0.02)]=0.5
            c_new[colors>0.02]=1
            colors=c_new
        # colors-=colors.min()
        # colors/=colors.max()
        colors=cm.plasma(colors)
        
            # def color_choice(r: float):
            #     if r==0.1:
            #         return "tab:blue"
            #     elif r==0.5:
            #         return "firebrick"
            #     else:
            #         return "gold"
            # colors=[color_choice(r) for r in colors]
    else:
        colors="w"
    ax.add_collection(collections.LineCollection(edges, linewidths=radii, colors=colors, antialiaseds=True, capstyle="round"))
    figure.canvas.draw()
    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    plt.close(figure)

    if colorize:
        img_gray = np.array(img.astype(np.float32))
    else:
        img_gray = np.array(Image.fromarray(img).convert("L")).astype(np.uint16)
    return img_gray, blackdict

def rasterize_forest_3D(forest: dict, image_scale_factor: np.ndarray, radius_list:list=None, min_radius=0, max_radius=1, blackdict: dict[str, bool]=None):
    # initialize canvas with defined image dimensions
    size_x = 1
    image_dim = tuple([math.ceil(image_scale_factor * d) for d in [76,76]])
    no_pixels_x, no_pixels_y = image_dim
    voxel_size_x = size_x / (no_pixels_x)
    dpi = 100
    x_inch = no_pixels_x*1.8 / dpi
    y_inch = no_pixels_y*1.8 / dpi
    plt.style.use('dark_background') # Dark theme
    figure = plt.figure(figsize=(x_inch,y_inch))
    figure.patch.set_facecolor('black')
    ax = plt.axes([0., 0., 1., 1], xticks=[], yticks=[], projection='3d', elev=50, azim=10)
    # ax.invert_yaxis()
    # Make panes transparent
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane

    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    # Remove the axis
    plt.axis('off')

    edges = []
    radii = []
    if blackdict is None:
        blackdict = dict()
        # p = random.random()**10 * 0.02
        p=0
    else:
        p=0
    for edge in forest:
        radius = float(edge["radius"])
        if radius<min_radius or radius>max_radius:
            continue
        radius *= 1.3
        current_node = edge["node1"]
        proximal_node = edge["node2"]
        if isinstance(current_node, np.ndarray) or isinstance(current_node, list):
            current_node = tuple(current_node)
            proximal_node = tuple(proximal_node)
        elif isinstance(current_node, str):
            # Legacy
            current_node = tuple([float(coord) for coord in current_node[1:-1].split(" ") if len(coord)>0])
            proximal_node = tuple([float(coord) for coord in proximal_node[1:-1].split(" ") if len(coord)>0])

        if proximal_node in blackdict or random()<p:
            blackdict[current_node] = True
            continue
        radius_list.append(radius)
        thickness = radius / voxel_size_x
        edges.append([(current_node[0],current_node[1],current_node[2]),(proximal_node[0],proximal_node[1],proximal_node[2])])
        radii.append(thickness)
    ax.add_collection(Line3DCollection(edges, linewidths=radii, colors="w", antialiaseds=True, capstyle="round"))
    figure.canvas.draw()
    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    plt.close(figure)
    img_gray = np.array(Image.fromarray(img).convert("L"))[195*4:-40*4,70*4:-70*4]
    return img_gray.astype(np.uint16)

def getCrossSlice(p1: tuple[int], p2: tuple[int], radius: int, voxel_size=1, image_dim=(255, 251, 120), mode: Literal['tube', 'cuboid']='cuboid'):
    """
    Computes relevant indices in an image tensor that contain the line from p1 to p2 with the given radius.
    
    Paramters:
        - p1: 3D point in simulation space
        - p2: 3D point in simulation space
        - radius: radius of line in simulation space scale
        - image_dim: shape of image tensor in voxels
        - mode: Type of indexing strategy that being used. 'tube' is more precise and better for long lines. 'cuboid' is faster to compute and better for short lines
    """
    if mode=='tube':
        p1_scaled = p1/voxel_size
        p1_voxel = np.floor(p1_scaled)
        dims = range(len(p1_voxel))
        line = (p2-p1)
        num_steps = np.linalg.norm(line) / voxel_size
        step_update = line / np.linalg.norm(line)
        voxel_offset = math.ceil(radius / voxel_size)
        indices = list()
        current_indices = np.array(list(itertools.product(*[
            [p1_scaled[d] + k for k in range(-voxel_offset, voxel_offset+2)] for d in dims
        ])))
        indices = {tuple(vox) for vox in floor(current_indices)}

        for i in range(math.ceil(num_steps)):
            current_indices = current_indices + step_update
            indices.update([tuple(vox) for vox in floor(current_indices)])

        indices = np.array(list(indices))
        indices = indices[np.all(indices>0, axis=1)]
        indices = indices[np.all(image_dim-indices>0, axis=1)]
        return indices

    if mode=='cuboid':
        voxel_offset = (radius / voxel_size) * math.sqrt(2)
        s_x,s_y,s_z = p1/voxel_size
        e_x,e_y,e_z = p2/voxel_size
        if s_x>e_x:
            e_x, s_x = s_x, e_x
        if s_y>e_y:
            e_y, s_y = s_y, e_y
        if s_z>e_z:
            e_z, s_z = s_z, e_z
        s_x = max(0, math.floor(s_x-voxel_offset))
        e_x = min(image_dim[0],math.ceil(e_x+voxel_offset+1))
        s_y = max(0, math.floor(s_y-voxel_offset))
        e_y = min(image_dim[1],math.ceil(e_y+voxel_offset+1))
        s_z = max(0, math.floor(s_z-voxel_offset))
        e_z = min(image_dim[2],math.ceil(e_z+voxel_offset+1))
        indices = np.array(list(itertools.product(
            list(range(s_x, e_x)),
            list(range(s_y, e_y)),
            list(range(s_z, e_z))
        )))
        return indices

    raise NotImplementedError(mode)

def voxelize_forest(forest: dict, image_scale_factor: np.ndarray, radius_list:list=None, min_radius=0, max_radius=1):
    size_x = 1
    image_dim = tuple([max(30,math.ceil(image_scale_factor * d)) for d in [76,76,1]])
    no_voxel_x, no_voxel_y, no_voxel_z = image_dim
    voxel_size_x = size_x / (no_voxel_x)
    pos_correction = np.array([0,0,10*voxel_size_x])
    voxel_diag = np.linalg.norm(np.array([voxel_size_x, voxel_size_x, voxel_size_x]))

    # img = np.zeros((no_voxel_x, no_voxel_y), np.uint8)
    img = np.zeros((no_voxel_x, no_voxel_y, no_voxel_z))
    # for tree in tqdm(forest.get_trees(), desc="Voxelizing forest"):
    #     for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False):
            # proximal_node = current_node.get_proximal_node()
            # radius = current_node.radius
    for edge in tqdm(forest, desc="Voxelizing forest"):
        radius = float(edge["radius"])
        if radius<min_radius or radius>max_radius:
            continue
        current_node = edge["node1"]
        proximal_node = edge["node2"]
        if not isinstance(edge["node1"], np.ndarray):
            current_node = [float(coord) for coord in edge["node1"][1:-1].split(" ") if len(coord)>0]
            proximal_node = [float(coord) for coord in edge["node2"][1:-1].split(" ") if len(coord)>0]
        radius_list.append(radius)

        voxel_indices = np.array(getCrossSlice(
            current_node+pos_correction, proximal_node+pos_correction, radius,voxel_size_x, image_dim
        ))
        if len(voxel_indices) == 0:
            continue
        indices = (voxel_indices+.5) * voxel_size_x

        # Calculate orthogonal projection of each voxel onto segment
        segment_vector = (current_node+pos_correction) - (proximal_node+pos_correction)
        voxel_vector = indices - (proximal_node+pos_correction)
        scalar_projection = np.dot(voxel_vector, segment_vector) / np.dot(segment_vector, segment_vector)
        inside_segment = np.logical_and(scalar_projection > 0, scalar_projection < 1)

        # If the projection falls onto the segment, add the vessel's contribution to the oxygen map
        vector_projection = (proximal_node+pos_correction) + np.dot(scalar_projection[:, None], segment_vector[None, :])
        dist = np.linalg.norm(indices - vector_projection, axis=1)

        inds: list[list] = voxel_indices[inside_segment].astype(np.uint16).transpose().tolist()
        volume_contribution = 1 - ((dist[inside_segment] - (radius - voxel_diag/2)) / voxel_diag)
        
        img[tuple(inds)] = np.maximum(volume_contribution, img[tuple(inds)])
        # Handle beginning and end
        dist = np.minimum(
            np.linalg.norm(indices-(current_node+pos_correction), axis=1),
            np.linalg.norm(indices-(proximal_node+pos_correction), axis=1)
        )
        inds = voxel_indices.astype(np.uint16).transpose().tolist()
        img[tuple(inds)] = np.maximum(1-((dist - (radius - voxel_diag/2)) / voxel_diag), img[tuple(inds)])
    # img[img>0]=1
    img = (255*np.clip(img,0,1))
    return img.astype(np.uint16)

def save_2d_projections(img: np.ndarray, out_dir, prefix: str, *, dims=(0,1,2)):
    for dim in dims:
        img_proj = np.max(img, dim)
        Image.fromarray(img_proj.astype(np.uint8)).save(f'{out_dir}/{prefix}.png')

def save_2d_img(img: np.ndarray, out_dir: str,  prefix: str):
    Image.fromarray(img.astype(np.uint8)).save(f'{out_dir}/{prefix}.png')

def plot_vessel_radii(out_dir: str, radius_list: list = []):
    plt.figure()
    bins = np.linspace(min(radius_list), max(radius_list),40)
    plt.xlim([min(radius_list), max(radius_list)])

    plt.hist(radius_list, bins=bins, alpha=0.5)
    plt.title('Vessel Radii Distribution')
    plt.xlabel('Radius')
    plt.ylabel('Count')
    plt.gca().set_yscale('log')

    plt.savefig(f"{out_dir}/hist.png", bbox_inches="tight")
    plt.close()

def forest_to_edges(forest: Forest):
    edges = [{
            "node1": current_node.position,
            "node2": current_node.get_proximal_node().position,
            "radius": current_node.radius
        } for tree in forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]
    return edges
