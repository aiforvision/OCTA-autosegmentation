import torch
import random
from PIL import Image
from models.noise_model import NoiseModel
import numpy as np
import csv 

from monai.transforms import *
from monai.config import KeysCollection
from vessel_graph_generation.tree2img import rasterize_forest
from models.networks import MODEL_DICT

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

class ImageToImageTranslationd(MapTransform):
    """
    Use a pre-trained GAN to transform a synthetic image into the real domain
    """
    def __init__(self, model_path: str, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.model: torch.nn.Module = MODEL_DICT["resnetGenerator9"]()
        checkpoint_G = torch.load(model_path)
        self.model.load_state_dict(checkpoint_G['model'])

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
        blackdict = None
        for i, key in enumerate(self.keys):
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
        downsample_factor=4
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
                d = self.noise_model.forward(img.unsqueeze(0), background, False, downsample_factor=self.downsample_factor).squeeze(0).detach()
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

def get_data_augmentations(aug_config: list[dict], dtype=torch.float32) -> list:
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
        augs.append(aug(**aug_d))
    return augs