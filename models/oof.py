import numpy as np
from numpy.fft import fftn, ifftn
from scipy.special import jv as besselj
import torch

EPSILON = 1e-12

class OOF:
    """
    2D Optimal Oriented Flux (OOF) filter.
    Code based on https://github.com/fepegar/optimally-oriented-flux
    """
    def __init__(self, **kwargs):
        self.nifti = None
        self.radii = None

        self.spacing = 1, 1
        self.num_radii = 5
        self.radii = self.get_radii()

        self.sigma = min(self.spacing)

        self.response_type = 1
        self.use_absolute = True
        self.normalization_type = 1

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the frangi filter image. Binarization needs to be applied afterwards.
        Implementation based on https://de.mathworks.com/matlabcentral/fileexchange/41612-optimally-oriented-flux-oof-for-3d-curvilinear-structure-detection

        Args:
            img (torch.Tensor): Image tensor of shape [B,C,H,W] scaled to [0,1]

        Returns:
            torch.Tensor: filtered image tensor of shape [B,C,H,W] scaled to [0,1]
        """
        assert img.shape[0]==1
        oof = img.squeeze().numpy() * 255
        oof = self.compute_oof(oof, self.radii)
        oof = oof+oof.max()
        oof = oof / oof.max()
        oof = torch.tensor(oof).view(img.shape)
        return oof

    def eval(self):
        pass
    def train(self):
        pass

    def get_radii(self) -> np.ndarray:
        return np.arange(1, self.num_radii + 1) * min(self.spacing)

    def check_normalization(self, radii):
        if min(radii) < self.sigma and self.normalization_type > 0:
            print('Sigma must be >= minimum range to enable the advanced'
                  ' normalization. The current setting falls back to'
                  ' normalization_type = 0 because of the undersize sigma.')
            self.normalization_type = 0

    def compute_oof(self, array: np.ndarray, radii: np.ndarray) -> np.ndarray:
        array = array.astype(np.double)
        shape = array.shape
        output = np.zeros(shape)
        self.check_normalization(radii)
        imgfft = fftn(array)
        x, y, sphere_radius = get_min_sphere_radius(shape, self.spacing)

        for radius in radii:
            # tqdm.write(f'Computing radius {radius:.3f}...')
            circle = circle_length(radius)
            nu = 1.5
            z_circle = circle * EPSILON
            bessel = besselj(nu, z_circle) / EPSILON**(3 / 2)
            base = radius / np.sqrt(2 * radius * self.sigma - self.sigma**2)
            exponent = self.normalization_type
            volume = get_circle_area(radius)
            normalization = volume / bessel / radius**2 * base**exponent

            exponent = - self.sigma**2 * 2 * np.pi**2 * sphere_radius**2
            num = normalization * np.exp(exponent)
            den = sphere_radius**(3/2)
            besselj_buffer = num / den

            cs = circle * sphere_radius
            a = np.sin(cs) / cs - np.cos(cs)
            b = np.sqrt(1 / (np.pi**2 * radius * sphere_radius))
            besselj_buffer = besselj_buffer * a * b * imgfft

            outputfeature_11 = np.real(ifftn(x * x * besselj_buffer))
            outputfeature_12 = np.real(ifftn(x * y * besselj_buffer))
            outputfeature_22 = np.real(ifftn(y * y * besselj_buffer))

            eigenvalues = np.linalg.eigvals(np.array([[outputfeature_11,outputfeature_12], [outputfeature_12, outputfeature_22]]).transpose([2,3,0,1]))

            lambda_1, lambda_2 = eigenvalues[:,:,0], eigenvalues[:,:,1]
            # The following code sorts the unorderred eigenvalues according to their magnitude
            maxe = np.copy(lambda_1)
            mine = np.copy(lambda_1)
            # mide = lambda_1 + lambda_2 + lambda_3
            mide = lambda_1 + lambda_2

            if self.use_absolute:
                maxe[np.abs(lambda_2) > np.abs(lambda_1)] = lambda_2[np.abs(lambda_2) > np.abs(lambda_1)]
                mine[np.abs(lambda_2) < np.abs(mine)] = lambda_2[np.abs(lambda_2) < np.abs(mine)]

                # maxe[np.abs(lambda_3) > np.abs(maxe)] = lambda_3[np.abs(lambda_3) > np.abs(maxe)]
                # mine[np.abs(lambda_3) < np.abs(mine)] = lambda_3[np.abs(lambda_3) < np.abs(mine)]
            else:
                # raise NotImplementedError
                maxe[lambda_2 > np.abs(maxe)] = lambda_2[lambda_2 > np.abs(maxe)]
                mine[lambda_2 < np.abs(mine)] = lambda_2[lambda_2 < np.abs(mine)]

                # maxe[lambda_3 > np.abs(maxe)] = lambda_3[lambda_3 > np.abs(maxe)]
                # mine[lambda_3 < np.abs(mine)] = lambda_3[lambda_3 < np.abs(mine)]

            mide -= maxe + mine

            if self.response_type == 0:
                tmpfeature = maxe
            elif self.response_type == 1:
                tmpfeature = maxe + mide
            elif self.response_type == 2:
                tmpfeature = np.sqrt(np.maximum(0, maxe * mide))
            elif self.response_type == 3:
                tmpfeature = np.sqrt(
                    np.maximum(0, maxe * mide) * np.maximum(0, mide))
            elif self.response_type == 4:
                tmpfeature = np.maximum(0, maxe)
            elif self.response_type == 5:
                tmpfeature = np.maximum(0, maxe + mide)
            else:
                raise NotImplementedError

            stronger_response = np.abs(tmpfeature) > np.abs(output)
            output[stronger_response] = tmpfeature[stronger_response]
        return output

def get_min_sphere_radius(shape, spacing):
    x, y = ifft_shifted_coordinates_matrix(shape)
    si, sj = shape
    pi, pj = spacing
    x /= si * pi
    y /= sj * pj
    sphere_radius = np.sqrt(x**2 + y**2) + EPSILON
    return x, y, sphere_radius
    
def get_circle_area(radius):
    return np.pi * radius**2

def circle_length(radius):
    return 2 * np.pi * radius

def ifft_shifted_coordinates_matrix(shape):
    shape = np.array(shape)
    dimensions = len(shape)
    p = shape // 2
    result = []

    for i in range(dimensions):
        x = np.arange(p[i], shape[i])
        y = np.arange(p[i])
        a = np.concatenate((x, y)) - p[i]
        reshapepara = np.ones(dimensions, np.uint16)
        reshapepara[i] = shape[i]
        A = np.reshape(a, reshapepara)
        repmatpara = np.copy(shape)
        repmatpara[i] = 1
        coords = np.tile(A, repmatpara).astype(float)
        result.append(coords)
    return result
