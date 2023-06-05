import argparse

from vessel_graph_generation.forest import Forest
from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.utilities import prepare_output_dir, read_config
import vessel_graph_generation.tree2img as tree2img
import numpy as np
import os
from tqdm import tqdm
import random
import yaml


def main(_):
    # Read config file
    config = read_config(args.config_file)
    # Initialize greenhouse
    greenhouse = Greenhouse(config['Greenhouse'])
    # Prepare output directory
    out_dir = prepare_output_dir(config['output'])
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Initialize forest
    arterial_forest = Forest(config['Forest'], greenhouse.simspace)
    venous_forest = Forest(config['Forest'], greenhouse.simspace, arterial=False)#, template_forest=arterial_forest)
    
    greenhouse.set_forests(arterial_forest, venous_forest)

    greenhouse.develop_forest()
    if config["output"]["save_stats"]:
        greenhouse.save_stats()

    image_scale_factor = config['output']['image_scale_factor']
    radius_list=[]

    if config["output"].get("save_color"):
        colors = [
            [1,0,0], #red
            [1,0.5,0], #orange
            [1,1,0], #yellow
            [0.5,1,0], #lime
            [0,1,0], #green
            [0,1,0.5], #green-blue
            [0,1,1], #cyan
            [0,0.5,1], #cyan-blue
            [0,0,1], #blue
            [0.5,0,1], #blue-pink
            [1,0,1], #pink
            [1,0,0.5], #magenta
            [0.5,0,0.5], #puple
            [0,0.39,0], #dark-green
            [0.55,0.27,0.07], #saddle-brown
            [0.98, 0.5,0.45] # salmon
        ]
        random.shuffle(colors)

        forest_mat = 0
        for tree in arterial_forest.get_trees() + venous_forest.get_trees():
            edges = [{
                "node1": current_node.position,
                "node2": current_node.get_proximal_node().position,
                "radius": current_node.radius
            } for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]
            
            tree_mat = tree2img.voxelize_forest(edges, image_scale_factor, radius_list, min_radius=0)
            tree_mat = np.tile(tree_mat[:,:,np.newaxis], 3).astype(np.float64)
            color = np.tile(np.array([[colors.pop()]]), (*tree_mat.shape[:-1], 1))
            tree_mat = tree_mat * color
            forest_mat = np.maximum(forest_mat, tree_mat)
        tree2img.save_2d_img(forest_mat, out_dir, "art_ven_color")


    art_edges = [{
        "node1": current_node.position,
        "node2": current_node.get_proximal_node().position,
        "radius": current_node.radius
    } for tree in arterial_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

    if venous_forest is not None:
            ven_edges = [{
            "node1": current_node.position,
            "node2": current_node.get_proximal_node().position,
            "radius": current_node.radius
        } for tree in venous_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

    if config['output']['save_trees']:
        import csv
        name = out_dir.split("/")[-1]
        filepath = os.path.join(out_dir, name+'.csv')
        with open(filepath, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["node1", "node2", "radius"])
            for row in art_edges+ven_edges:
                writer.writerow([row["node1"], row["node2"], row["radius"]])

    if config["output"]["save_3D_volumes"]:
        art_mat = tree2img.voxelize_forest(edges, image_scale_factor, radius_list, min_radius=0)
        if config["output"]["save_art"]:
            np.save(f'{out_dir}/art_img.npy', art_mat)

        art_mat = tree2img.voxelize_forest(edges, image_scale_factor, radius_list, min_radius=0)
        
        if config["output"]["save_art"]:
            tree2img.save_2d_projections(art_mat, out_dir, 'art_img', config["output"])
            if config["output"]["save_3D_volumes"]:
                np.save(f'{out_dir}/art_img.npy', art_mat)
        
        if venous_forest is not None:
            edges = [{
            "node1": current_node.position,
            "node2": current_node.get_proximal_node().position,
            "radius": current_node.radius
        } for tree in venous_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]
            ven_mat = tree2img.voxelize_forest(edges, image_scale_factor, radius_list, min_radius=0)
            if config["output"]["save_ven"]:
                np.save(f'{out_dir}/ven_img.npy', art_mat)
        else:
            ven_mat = 0

        if config["output"]["save_art_ven_gray"]:
            art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
            np.save(f'{out_dir}/art_ven_img_gray.npy', art_ven_mat_gray)

        if config["output"]["save_art_ven_rgb"]:
            art_mat = np.tile(art_mat[:,:,:,np.newaxis], 3).astype(np.float64)
            red = np.tile(np.array([[[[1,0,0]]]]), (*art_mat.shape[:-1], 1))
            art_mat = art_mat * red

            ven_mat = np.tile(ven_mat[:,:,:,np.newaxis], 3).astype(np.float64)
            blue = np.tile(np.array([[[[0,0,1]]]]), (*ven_mat.shape[:-1], 1))
            ven_mat = ven_mat * blue

            art_ven_mat = np.maximum(art_mat, ven_mat)
            art_ven_mat = np.clip(art_ven_mat, 0, 255).astype(np.uint8)
            np.save(f'{out_dir}/art_ven_img_rgb.npy', art_ven_mat)

    else:
        art_mat = tree2img.rasterize_forest(art_edges, image_scale_factor, radius_list, min_radius=0)[0]
        if config["output"]["save_art"]:
            tree2img.save_2d_img(art_mat, out_dir, "art_img")

        if venous_forest is not None:
            ven_mat = tree2img.rasterize_forest(ven_edges, image_scale_factor, radius_list, min_radius=0)[0]
            if config["output"]["save_ven"]:
                tree2img.save_2d_img(ven_mat, out_dir, "ven_img")
        else:
            ven_mat = 0

        if config["output"]["save_art_ven_gray"]:
            art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)
            tree2img.save_2d_img(art_ven_mat_gray, out_dir, "art_ven_img_gray")
        
        if config["output"]["save_art_ven_rgb"]:
            art_mat = np.tile(art_mat[:,:,np.newaxis], 3).astype(np.float64)
            red = np.tile(np.array([[[[1,0,0]]]]), (*art_mat.shape[:-1], 1))
            art_mat = art_mat * red

            ven_mat = np.tile(ven_mat[:,:,np.newaxis], 3).astype(np.float64)
            blue = np.tile(np.array([[[[0,0,1]]]]), (*ven_mat.shape[:-1], 1))
            ven_mat = ven_mat * blue

            art_ven_mat = np.maximum(art_mat, ven_mat)
            art_ven_mat = np.clip(art_ven_mat, 0, 255).astype(np.uint8)
            tree2img.save_2d_img(ven_mat, out_dir, "art_ven_rgb")


    if config["output"]["save_stats"]:
        tree2img.plot_vessel_radii(out_dir, radius_list)

    # os.chdir(os.path.abspath('../..'))

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--threads', help="Number of parallel threads", type=int, default=-1)
    args = parser.parse_args()

    if args.debug:
        import warnings
        warnings.filterwarnings('error')

    # from multiprocessing import cpu_count
    # from multiprocessing.dummy import Pool
    # from multiprocessing.pool import ThreadPool
    # if args.threads == -1:
    #     cpus = cpu_count()
    #     threads = cpus-1 if cpus>1 else 1
    # else:
    #     threads=args.threads
    # if threads>1:
    #     pool: ThreadPool = Pool(threads)
    #     results = list(tqdm(pool.imap(main, [args.config_file for _ in range(args.num_samples)]), total=args.num_samples, desc="Generating vessel graph..."))
    # else:
    for i in tqdm(range(args.num_samples), desc="Generating vessel graph..."):
        main(args.config_file)
