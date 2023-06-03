import datetime
# import json
import logging
import os

import numpy as np
import math
import yaml


# Function to set up logging based on python's logging module
def configure_logging():

    logpath = os.path.join(os.getcwd(), 'log.txt')
    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(filename=logpath, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info('Logging to {}.'.format(logpath))


def normalize_vector(vec):

    if np.linalg.norm(vec) != 0.0:
        return vec / np.linalg.norm(vec)
    else:
        return vec


# Function to prepare output directory
def prepare_output_dir(config):

    output_dir = os.path.abspath(config['directory'])
    if config['subdir_with_timestamp']:
        output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    logging.info('{} - Set output directory to {}.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3], output_dir))


# Function to read in JSON config file
def read_config(configpath):

    path = os.path.abspath(configpath)
    with open(path, "r") as filepath:
    #     config = json.load(filepath)
        config: dict[str,dict] = yaml.safe_load(filepath)

    # Using print as logger is not configured yet

    print('Read config file {}.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3], filepath))

    return config

def eukledian_dist(pos1: tuple[float], pos2: tuple[float]) -> float:
    dist = [(a - b)**2 for a, b in zip(pos1, pos2)]
    dist = math.sqrt(sum(dist))
    return dist
    # return np.linalg.norm(np.array(pos1)-np.array(pos2))

def norm_vector(v: tuple[float]) -> float:
    return v/np.linalg.norm(v)

def get_angle_between_vectors(u, V):
    # https://stackoverflow.com/a/2827466
    C = np.dot(u,V.T)/np.linalg.norm(u)/np.linalg.norm(V,axis=1) # cosine of the angles
    angles = np.degrees(np.arccos(np.minimum(np.maximum(C, -1), 1))) # angles
    return angles

def get_angle_between_two_vectors(u, v):
    # https://stackoverflow.com/a/2827466
    c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
    angle = np.degrees(np.arccos(min(max(c, -1), 1))) # if you really want the angle
    return angle
