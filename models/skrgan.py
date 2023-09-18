import torch
import numpy as np
from skimage.morphology import area_opening, area_closing
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter

class SkrGAN:
    def __init__(self, sigma=2, area_threshold_open=64, connectivity_open=1, area_threshold_close=64, connectivity_close=1) -> None:
        self.sigma = sigma
        self.area_threshold_open = area_threshold_open
        self.connectivity_open = connectivity_open
        self.area_threshold_close = area_threshold_close
        self.connectivity_close = connectivity_close

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_numpy: np.ndarray = img.detach().cpu().numpy().astype(np.float32).squeeze()

        # Sobel edge detection
        sobel_h = sobel(img_numpy, 0)  # horizontal gradient
        sobel_v = sobel(img_numpy, 1)  # vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        magnitude -= magnitude.min()
        magnitude /= magnitude.max()

        # Gaussian low pass filter
        filtered_image = gaussian_filter(magnitude, sigma=self.sigma)

        opened_image = area_opening(filtered_image, area_threshold=64, connectivity=1)
        opened_image -= opened_image.min()
        opened_image /= opened_image.max()
        closed_image = area_closing(opened_image, area_threshold=64, connectivity=1)
        closed_image -= closed_image.min()
        closed_image /= closed_image.max()
        return torch.tensor(closed_image).unsqueeze(0).unsqueeze(0)

    def eval(self):
        pass
    def train(self):
        pass