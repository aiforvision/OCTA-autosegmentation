import csv
import pickle
import random
from typing import Tuple

import numpy as np
import torch
from models.networks import MODEL_DICT
from models.noise_model import NoiseModel
from monai.config import KeysCollection
from monai.transforms import *  # noqa: F403
from monai.transforms import (
    KeepLargestConnectedComponent,
    MapTransform,
    Randomizable,
    Transform,
)
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import line
from utils.enums import Phase
from vessel_graph_generation.tree2img import rasterize_forest


class SpeckleBrightnesd(MapTransform):
    """
    Speckle noise component of our noise model. Randomly decrease the brightness of the image in random areas.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            c = torch.rand((1,1,9,9))*0.5+0.5
            C = torch.nn.functional.interpolate(c, size=img.shape[-2:], mode="bilinear").squeeze(0)
            R = C - (torch.rand_like(C) * (1-C))
            img = img*R
            img /= img.max()
            img -= img.min()
            data[key] = img
        return data
    
class BinomialVesselNoised(MapTransform):
    """
    Binomial noise component of our noise model. Randomly add vessel-like noise to the image.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, vessel_noise_scaling = 0.5, vessel_noise_blur = 1.0, r=48) -> None:
        super().__init__(keys, allow_missing_keys)
        self.vessel_noise_scaling = vessel_noise_scaling
        self.vessel_noise_blur = vessel_noise_blur
        self.r = r

    def add_noise(self, img: np.ndarray, vessel_noise_scaling = 0.5, vessel_noise_blur = 1.0, r=48) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        - img: Numpy array of 2d grayscale image
        - vessel_noise_scaling (float): Scaling factor for vessel noise
        - vessel_noise_blur (float): Blurring factor for vessel noise
        - r (int): Radius of the vessel noise

        Returns:
        --------
        - img: Augmented numpy array of 2d grayscale image
        """
        # Vessel noise
        vessel_noise = np.random.binomial(1, 0.1, size=img.shape)
        vessel_noise = binary_dilation(vessel_noise, iterations=1).astype(float)

        for i in range(vessel_noise.shape[0]):
            for j in range(vessel_noise.shape[1]):
                if np.sqrt((i - vessel_noise.shape[0]/2) ** 2 + (j - vessel_noise.shape[1]/2) ** 2) < r:
                    vessel_noise[i, j] = vessel_noise[i, j] * 0.7
                if np.sqrt((i - vessel_noise.shape[0]/2) ** 2 + (j - vessel_noise.shape[1]/2) ** 2) < r - 3:
                    vessel_noise[i, j] = vessel_noise[i, j] * 0.7
                if np.sqrt((i - vessel_noise.shape[0]/2) ** 2 + (j - vessel_noise.shape[1]/2) ** 2) < r - 6:
                    vessel_noise[i, j] = vessel_noise[i, j] * 0.7
                if np.sqrt((i - vessel_noise.shape[0]/2) ** 2 + (j - vessel_noise.shape[1]/2) ** 2) < r - 9:
                    vessel_noise[i, j] = vessel_noise[i, j] * 0.7
                if np.sqrt((i - vessel_noise.shape[0]/2) ** 2 + (j - vessel_noise.shape[1]/2) ** 2) < r - 12:
                    vessel_noise[i, j] = vessel_noise[i, j] * 0.7

        vessel_noise = gaussian_filter(vessel_noise, vessel_noise_blur) * vessel_noise_scaling

        # Quantum noise
        quantum_noise_scale = 0.2
        quantum_noise = np.random.uniform(0.0, quantum_noise_scale, size=img.shape)

        # Add noth noise sources to image
        img = np.clip((img + vessel_noise + quantum_noise) / (1.0 + vessel_noise_scaling/1.5), 0.0, 1.0)
        return img

    def __call__(self, data):
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                img: torch.Tensor = data[key]
                img_shape = img.shape
                img = img.squeeze().numpy()
                img = self.add_noise(img, self.vessel_noise_scaling, self.vessel_noise_blur, self.r)
                data[key] = torch.tensor(img).view(img_shape)
        return data
    
class AddVitreousFloater(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False,
            floater_chance: float = 0.1,
            floater_opacity_interval: tuple[float,float]=(0.5, 1.0),
            floater_segments_interval: tuple[int,int]=(10, 20),
            dilations_interval: tuple[int,int] = (10, 30)) -> None:
        super().__init__(keys, allow_missing_keys)
        self.floater_chance = floater_chance
        self.floater_opacity_interval = floater_opacity_interval
        self.floater_segments_interval = floater_segments_interval
        self.dilations_interval = dilations_interval

    def __call__(self, data):
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                img: torch.Tensor = data[key]
                img_shape = img.shape
                img = img.squeeze().numpy()
                img = self.add_vitreous_floater(img, self.floater_chance, self.floater_opacity_interval, self.floater_segments_interval, self.dilations_interval)
                data[key] = torch.tensor(img).view(img_shape)
        return data

    def add_vitreous_floater(self,
            img: np.ndarray,
            floater_chance: float = 0.1,
            floater_opacity_interval: tuple[float,float]=(0.5, 1.0),
            floater_segments_interval: tuple[int,int]=(10, 20),
            dilations_interval: tuple[int,int] = (10, 30)
        ) -> np.ndarray:
        """
        Parameters:
        -----------
        - img: Numpy array of 2d grayscale image
        - floater_chance (float): Chance of adding a floater
        - floater_opacity_interval (tuple): Interval for floater opacity
        - floater_segments_interval (tuple): Interval for floater segments
        - dilations_interval (tuple): Interval for floater dilations

        Returns:
        --------
        - img: Augmented numpy array of 2d grayscale image
        """
        # Vitreous floater artifacts
        if np.random.uniform() < floater_chance:
            size_x = img.shape[1]
            size_y = img.shape[0]

            floater = np.zeros((size_x, size_y))

            starting_x = np.random.randint(0, size_x)
            starting_y = np.random.randint(0, size_y)
            current_point = np.array((starting_x, starting_y))

            points = []
            points.append(current_point)

            floater_opacity = np.random.uniform(*floater_opacity_interval)

            floater_segments = np.random.randint(*floater_segments_interval)

            for i in range(floater_segments):

                dx = int(np.random.normal(scale=size_x / 10))
                dy = int(np.random.normal(scale=size_y / 10))
                next_point = current_point + (dx, dy)

                rr, cc = line(current_point[0], current_point[1], next_point[0], next_point[1])

                inside_image = np.logical_and.reduce((rr >= 0, rr < size_x, cc >= 0, cc < size_y))
                rr = rr[inside_image]
                cc = cc[inside_image]
                
                floater[rr, cc] = floater_opacity

                current_point = next_point

            dilations = np.random.randint(*dilations_interval)
            floater = binary_dilation(floater, iterations=dilations).astype(float)
            floater = gaussian_filter(floater, 10)

            img = img * (1 - floater)
        return img

class AddMotionArtifact(MapTransform):
    def __init__(self, img_key, gt_key,
           artifacts: dict[str,float] = {
                'shear': 0.3,
                'stretch': 0.3,
                'buckle': 0.3,
                'whiteout': 0.1
            },
            grace_margin: int = 10,
            max_shear: int = 5,
            max_stretch: int = 5,
            max_buckle: int = 5,
            max_whiteout: int = 1,
            no_h_cuts: int = 3) -> None:
        super().__init__([img_key, gt_key], False)
        self.img_key = img_key
        self.gt_key = gt_key
        self.artifacts = artifacts
        self.grace_margin = grace_margin
        self.max_shear = max_shear
        self.max_stretch = max_stretch
        self.max_buckle = max_buckle
        self.max_whiteout = max_whiteout
        self.no_h_cuts = no_h_cuts

    def __call__(self, data):
        img: torch.Tensor = data[self.img_key]
        img_shape = img.shape
        img = img.squeeze().numpy()

        gt: torch.Tensor = data[self.gt_key]
        gt_shape = gt.shape
        gt = gt.squeeze().numpy()

        img, gt = self.add_motion_artifact(img, gt, self.artifacts, self.grace_margin, self.max_shear, self.max_stretch, self.max_buckle, self.max_whiteout)
        
        data[self.img_key] = torch.tensor(img).view(img_shape)
        data[self.gt_key] = torch.tensor(gt).view(gt_shape)
        return data
    
    def add_motion_artifact(self,
            img: np.ndarray,
            gt: np.ndarray,
            artifacts: dict[str,float] = {
                'shear': 0.3,
                'stretch': 0.3,
                'buckle': 0.3,
                'whiteout': 0.1
            },
            grace_margin: int = 10,
            max_shear: int = 5,
            max_stretch: int = 5,
            max_buckle: int = 5,
            max_whiteout: int = 1,
            no_h_cuts: int = 3
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        - img: Numpy array of 2d grayscale image
        - gt: Numpy array of high resolution grayscale label
        - artifacts (dict): Dictionary of motion artifacts and their probabilities. Choose from 'shear', 'stretch', 'buckle', 'whiteout'
        - grace_margin (int): Margin around the image where no artifacts are added
        - max_shear (int): Maximum shear value
        - max_stretch (int): Maximum stretch value
        - max_buckle (int): Maximum buckle value
        - max_whiteout (int): Maximum whiteout value
        - no_h_cuts (int): Maximum number of horizontal cuts

        Returns:
        --------
        - img: Augmented numpy array of 2d grayscale image
        - gt: Adjusted numpy array of high resolution grayscale label
        """
        # Motion and decorrelation artifacts
        no_h_cuts = np.random.randint(0, no_h_cuts)

        for h_cut in range(no_h_cuts):

            temp_img = img.copy()
            temp_gt = gt.copy()
            artifact = np.random.choice(list(artifacts.keys()), p=list(artifacts.values()))
            position = np.random.randint(grace_margin, temp_img.shape[0] - grace_margin)

            if artifact == 'shear':
                shear = np.random.randint(0, max_shear + 1)
                img[:position, :] = temp_img[:position, :]
                img[position:, :] = np.roll(temp_img[position:, :], shear, axis=1)
                img[position:, :shear] = 0

                gt[:4*position, :] = temp_gt[:4*position, :]
                gt[4*position:, :] = np.roll(temp_gt[4*position:, :], 4*shear, axis=1)
                gt[4*position:, :4*shear] = 0

            elif artifact == 'stretch':
                stretch = np.random.randint(1, max_stretch + 1)
                img[:position, :] = temp_img[:position, :]
                img[position:position + stretch, :] = temp_img[position, :]
                img[position + stretch:, :] = temp_img[position:-stretch, :]

                gt[:4*position, :] = temp_gt[:4*position, :]
                gt[4*position:4*position + 4*stretch, :] = temp_gt[4*position, :]
                gt[4*position + 4*stretch:, :] = temp_gt[4*position:-4*stretch, :]

            elif artifact == 'buckle':
                buckle = np.random.randint(1, max_buckle + 1)
                img[:position, :] = temp_img[:position, :]
                img[position:, :] = temp_img[position-buckle:-buckle, :]

                gt[:4*position, :] = temp_gt[:4*position, :]
                gt[4*position:, :] = temp_gt[4*position-4*buckle:-4*buckle, :]

            elif artifact == 'whiteout':
                whiteout = np.random.randint(1, max_whiteout + 1)
                img[position:position + whiteout, :] = np.random.uniform(0.5, 1.0, size=(whiteout, temp_img.shape[1]))
        return img, gt

class MentenAugmentationd(MapTransform):
    """
    Applies Brightness, binomial noise, quantum noise, Vitreous floater artifacts, and motion artifacts as described in:

    Physiology-Based Simulation of the Retinal Vasculature Enables Annotation-Free Segmentation of OCT Angiographs
    Martin J. Menten, Johannes C. Paetzold, Alina Dima, Bjoern H. Menze, Benjamin Knier & Daniel Rueckert
    MICCAI 2022
    https://link.springer.com/chapter/10.1007/978-3-031-16452-1_32
    """
    def __init__(self, img_key: str, gt_key: str) -> None:
        super().__init__(keys=[img_key, gt_key], allow_missing_keys=False)
        self.img_key = img_key
        self.gt_key = gt_key
        self.binomialVesselNoised = BinomialVesselNoised([img_key], allow_missing_keys=True)
        self.vitreousFloater = AddVitreousFloater([img_key], allow_missing_keys=True)
        self.add_motion_artifact = AddMotionArtifact(img_key, gt_key)

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data = self.binomialVesselNoised(data)
        data = self.vitreousFloater(data)
        data = self.add_motion_artifact(data)
        return data

class ImageToImageTranslationd(MapTransform):
    """
    Use a pre-trained GAN to transform a synthetic image into the real domain
    """
    def __init__(self, model_path: str|dict[str,str], keys: KeysCollection, model_config: dict = None, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        if model_config is None:
            self.model: torch.nn.Module = MODEL_DICT["resnetGenerator9"]()
        else:
            self.model: torch.nn.Module = MODEL_DICT[model_config.pop("name")](phase=Phase.TEST, MODEL_DICT=MODEL_DICT,**model_config)
        if isinstance(model_path, dict):
            for k,v in model_path.items():
                checkpoint = torch.load(v)
                net: torch.nn.Module = getattr(self.model, k)
                net.load_state_dict(checkpoint["model"])
                print(f"Loaded network weights {k} from epoch {checkpoint['epoch']}.")
        else:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model'])
            print(f"Loaded network weights from epoch {checkpoint['epoch']}.")
        self.model.eval()


    def __call__(self, data):
        for key in self.keys:
            with torch.no_grad():
                img: torch.Tensor = data[key]
                img_t = self.model(img.float().unsqueeze(0)).squeeze(0)
                data[key] = img_t
        return data

class LoadGraphAndFilterByRandomRadiusd(MapTransform):
    """
    Given a graph csv file, only load edges with radius larger than the given threshold. Then, turn the graph into a grayscale image of the given shape.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, image_resolutions=[[304,304]], min_radius=[0], max_dropout_prob=0, MIP_axis = 2) -> None:
        super().__init__(keys, allow_missing_keys)
        self.min_radius = min_radius
        self.image_resolutions = image_resolutions
        self.max_dropout_prob = max_dropout_prob
        self.MIP_axis = MIP_axis

    def __call__(self, data):
        if "blackdict" in data:
            blackdict_path = data["blackdict"]
            with open(blackdict_path, mode="rb") as file:
                blackdict = pickle.load(file)
        else:
            blackdict = None
        for i, key in enumerate(self.keys):
            if key not in data and self.allow_missing_keys:
                continue
            f: list[dict] = list()
            with open(data[key], newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    f.append(row)
            img, blackdict = rasterize_forest(f, self.image_resolutions[i], self.MIP_axis, min_radius=self.min_radius[i], max_dropout_prob=self.max_dropout_prob, blackdict=blackdict)
            img_t = torch.tensor(img.astype(np.float32))
            data[key] = img_t
        return data

class ToGrayScaled(MapTransform):
    """
    Convert an RGB image into a grayscale image.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            if not self.allow_missing_keys or key in data:
                data[key] = torch.tensor(np.array(Image.fromarray(data[key].numpy().astype(np.uint8)).convert("L")).astype(np.float32))
        return data
    
class SelectSlice(MapTransform):
    """
    Select a slice from the given Tensor.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, slice_selection: list[int]=None) -> None:
        super().__init__(keys, allow_missing_keys)
        if slice_selection:
            self.slice_selection = tuple([slice(s,e) for s,e in slice_selection])

    def __call__(self, data):
        if self.slice_selection is not None:
            for key in self.keys:
                if not self.allow_missing_keys or key in data:
                    data[key] = data[key][self.slice_selection]
        return data
    
class RemoveOuterNoise(Transform):
    """
    Remove all components that are not connected to the central plane of the z-axis. Used primarily for 3D reconstruction post-processing.
    """
    def __init__(self, z_axis=0) -> None:
        super().__init__()
        self.keep_largest_connected_component = KeepLargestConnectedComponent()
        self.z_axis = z_axis

    def __call__(self, volume: torch.Tensor):
        volume_tmp = volume.clone().to(dtype=torch.bool, non_blocking=True).unsqueeze(0)
        volume_tmp[0,volume.shape[self.z_axis]//2].fill_(True)
        volume_tmp = self.keep_largest_connected_component(volume_tmp, ).squeeze(0)
        volume = torch.logical_and(volume,volume_tmp)
        return volume


class NoiseModeld(MapTransform):
    """
    Our proposed handcrafted noise model to simulate artifacts and contrast variations of real OCTA images.
    Given a synthetic vessel map I and a background vessel map I_D, the module successively performs 
        1) background noise addition,
        2) brightness augmentation, and
        3) contrast adaptation. 
    In each block, we use a sparse control point matrix to generate a field for locally varying contrast.

    The control point matrix can also be chosen adversarially. See /models/noise_model.py#L82
    """
    def __init__(self,
        keys: KeysCollection,
        prob=1,
        allow_missing_keys: bool = False,
        grid_size=(9,9),
        lambda_delta = 1,
        lambda_speckle = 0.7,
        lambda_gamma = 0.3,
        alpha=0.2,
        downsample_factor=1
    ) -> None:
        self.noise_model = NoiseModel(
            grid_size = grid_size,
            lambda_delta = lambda_delta,
            lambda_speckle = lambda_speckle,
            lambda_gamma = lambda_gamma,
            alpha=alpha
        )
        self.prob = prob
        self.downsample_factor = downsample_factor
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        if random.random()<self.prob:
            for key in self.keys:
                img: torch.Tensor = data[key]
                background: torch.Tensor = data["background"]
                d = self.noise_model.forward(img.unsqueeze(0), background.unsqueeze(0), False, downsample_factor=self.downsample_factor).squeeze(0).detach()
                data[key]=d
        return data

class RandomDecreaseResolutiond(MapTransform):
    """
    Randomly lower the Signal-to-noise ratio by downsamping the image by a random factor c and then upsample it again
    to the original size.
    """
    def __init__(self, keys: KeysCollection, p=1, max_factor=0.25) -> None:
        super().__init__(keys, True)
        self.max_factor = max_factor
        self.p = p

    def __call__(self, data):
        if random.uniform(0,1)<self.p:
            for key in self.keys:
                d: torch.Tensor = data[key]
                size = d.shape
                factor = random.uniform(self.max_factor,1)
                d = torch.nn.functional.interpolate(d.unsqueeze(0), scale_factor=factor)
                d = torch.nn.functional.interpolate(d, size=size[1:]).squeeze(0)
                data[key]=d
        return data

class AddRandomBackgroundNoised(MapTransform):
    """
    Add background noise to image. If there is no background noise image available, use random uniform noise.
    """
    def __init__(self, keys:  tuple[str], delete_background=True) -> None:
        super().__init__(keys, True)
        self.delete_background = delete_background

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                img: torch.Tensor = data[key]
                noise = data["background"] if "background" in data else torch.rand_like(img)
                speckle_noise = np.random.uniform(0,1,img.shape)
                img = torch.maximum(img, noise*speckle_noise)
                data[key] = img
        if self.delete_background and "background" in data:
            del data["background"]
        return data

class AddLineArtifact(MapTransform):
    """
    Generates a blurry horizontal line with is a common image artifact in OCTA images 
    """
    def __init__(self, keys: KeysCollection) -> None:
        """
        Generates a blurry horizontal line with is a common image artifact in OCTA images
        
        Parameters:
            - keys: List of dict keys where the artifact should be applied to
        """
        super().__init__(keys, False)
        self.c = torch.tensor([[0.0250, 0.0750, 0.3750, 0.8750, 1.0000, 0.8750, 0.3750, 0.0750, 0.0250]]).unsqueeze(-1)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            start = random.randint(0,img.shape[-2]-9)
            s = slice(start,start+9)
            line = img[:,s,:].unsqueeze(0)
            line = torch.conv2d(line, weight=torch.full((1,1,7,7), 1/50), padding="same")
            img[:,s,:] = img[:,s,:]*(1-self.c) + self.c * line[0,:,:,:]
            data[key] = img
        return data

class RandCropOrPadd(MapTransform):
    """
    Randomly crop or pad the image with a random zoom factor.
    """
    def __init__(self, keys: list[str], prob=0.1, min_factor=1, max_factor=1) -> None:
        """
        Randomly crop or pad the image with a random zoom factor.
        If zoom_factor > 1, the image will be zero-padded to fit the larger image shape.
        If zoom_factor > 1, the image will be cropped at a random center to fit the larger image shape.
        Parameters:
            - keys: List of dict keys where the noise should be applied to
            - prob: Probability with which the transform is applied
            - min_factor: Smallest allowed zoom factor
            - max_factor: Largest allowed zoom factor
        """
        super().__init__(keys)
        self.prob = prob
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, data):
        if random.uniform(0,1)<self.prob:
            factor = random.uniform(self.min_factor, self.max_factor)
            slice_x = slice_y = None
            for k in self.keys:
                d: torch.Tensor = data[k]
                if factor<1:
                    if slice_x is None:
                        s_x = int(d.shape[1]*factor)
                        s_y = int(d.shape[2]*factor)
                        start_x = random.randint(0, d.shape[1]-s_x)
                        start_y = random.randint(0, d.shape[2]-s_y)
                        slice_x = slice(start_x, start_x + s_x)
                        slice_y = slice(start_y, start_y + s_y)
                    d = d[:,slice_x, slice_y]
                elif factor>1:
                    frame = torch.zeros((d.shape[0], int(d.shape[1]*factor), int(d.shape[2]*factor)))
                    start_x = (frame.shape[1]-d.shape[1])//2
                    start_y = (frame.shape[2]-d.shape[2])//2
                    frame[:,start_x:start_x+d.shape[1], start_y:start_y+d.shape[2]] = d.clone()
                    d = frame
                data[k] = d
        return data

def get_data_augmentations(aug_config: list[dict], seed: int, dtype=torch.float32) -> list:
    if aug_config is None:
        return []
    augs = []
    for aug_d in aug_config:
        aug_d = dict(aug_d)
        aug_name: str = aug_d.pop("name")
        aug = globals()[aug_name]
        if aug_name.startswith("CastToType"):
            # Special handling for type to enable AMP training
            islist = isinstance(aug_d["dtype"], list)
            if not islist:
                aug_d["dtype"] = [aug_d["dtype"]]
            types = [dtype if t == "dtype" else getattr(torch, t) for t in aug_d["dtype"]]
            if islist:
                aug_d["dtype"] = types
            else:
                aug_d["dtype"] = types[0]
        if aug_name.startswith("Lambda"):
            aug_d["func"] = eval(aug_d["func"])
        aug_obj = aug(**aug_d)
        if isinstance(aug_obj, Randomizable):
            aug_obj.set_random_state(seed=seed)
        augs.append(aug_obj)
    return augs