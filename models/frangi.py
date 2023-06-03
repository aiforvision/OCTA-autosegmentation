from skimage.filters import frangi
import torch

class Frangi():
    def __init__(self, **kwargs) -> None:
        pass
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the frangi filter image. Binarization needs to be applied afterwards.
        Implementation uses https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi

        Args:
            img (torch.Tensor): Image tensor of shape [B,C,H,W] scaled to [0,1]

        Returns:
            torch.Tensor: filtered image tensor of shape [B,C,H,W] scaled to [0,1]
        """
        assert img.shape[0]==1
        frangi_img = img.squeeze().numpy() * 255
        frangi_img = frangi(frangi_img, sigmas = (0.5,2,0.5), alpha=1, beta=15, black_ridges=False)
        frangi_img = torch.tensor(frangi_img).view(img.shape)
        return frangi_img
    def eval(self):
        pass
    def train(self):
        pass
