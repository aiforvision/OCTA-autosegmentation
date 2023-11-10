from monai.losses import DiceLoss
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Union
from random import randint, uniform, choice
from torchvision.transforms.functional import rotate, InterpolationMode
from utils.enums import Phase

from models.noise_model import NoiseModel

class ANTLoss(torch.nn.Module):
    def __init__(self,
                 scaler: GradScaler,
                 loss_fun: torch.nn.Module,
                 grid_size=(9,9),
                 lambda_delta = 1,
                 lambda_speckle = 0.7,
                 lambda_gamma = 0.3,
                 max_decrease_res=0.25,
                 alpha=1e-3,
                 crop=(1,1),
                 label_threshold=0.1) -> None:
        super().__init__()
        self.noise_model = NoiseModel(
            grid_size = grid_size,
            lambda_delta = lambda_delta,
            lambda_speckle = lambda_speckle,
            lambda_gamma = lambda_gamma,
            alpha=alpha
        )
        self.scaler = scaler
        self.loss_fun = loss_fun
        self.crop = crop
        self.max_decrease_res = max_decrease_res
        self.decrease_res_factor = None
        self.label_threshold = label_threshold

    def _randomize_crop(self, sample: torch.Tensor):
        if self.crop[0] != 1 or self.crop[1] != 1:
            self.len_h = int(sample.shape[-2]*self.crop[0])
            self.len_w = int(sample.shape[-1]*self.crop[1])
            self.h_crop = [randint(0,sample.shape[-2]-self.len_h) for b in range(sample.shape[0])]
            self.w_crop = [randint(0,sample.shape[-1]-self.len_w) for b in range(sample.shape[0])]

    def _crop_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if self.crop[0] != 1 or self.crop[1] != 1:
            out = []
            for b in range(sample.shape[0]):
                out.append(sample[b,:,self.h_crop[b]:self.h_crop[b]+self.len_h,self.w_crop[b]:self.w_crop[b]+self.len_w])
            sample=torch.stack(out,dim=0)
        return sample
    
    def _rand_decrease_res(self, img: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(img.shape[0]):
            tmp = torch.nn.functional.interpolate(img[b:b+1], scale_factor=self.downsample_factor[b])
            out.append(torch.nn.functional.interpolate(tmp, size=img.shape[-2:])[0])
        return torch.stack(out, dim=0)
    
    def _rand_rotate(self, img: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(img.shape[0]):
            tmp = torch.rot90(img[b:b+1], self.rot_k[b], dims=(-2,-1))
            out.append(rotate(tmp, angle=self.rot_r[b], interpolation=InterpolationMode.BILINEAR)[0])
        return torch.stack(out, dim=0)
    
    def _create_adversarial_sample(self, x: torch.Tensor, background: torch.Tensor, y: torch.Tensor, adversarial: bool):
        with torch.cuda.amp.autocast():
            # out = []
            # for b in range(x.shape[0]):
            #     out.append(self.noise_model.forward(x[b:b+1], background[b:b+1], False, downsample_factor=1)[0])
            # adv_sample = torch.stack(out,dim=0)
            adv_sample = self.noise_model.forward(x, background, adversarial, downsample_factor=1)
            adv_sample = torch.nn.functional.interpolate(adv_sample, size=y.shape[-2:], mode="bilinear")
            adv_sample = self._rand_decrease_res(adv_sample)
            adv_sample = self._rand_rotate(adv_sample)
            adv_sample = self._crop_sample(adv_sample)
        return adv_sample

    def forward(self, model: torch.nn.Module, x: torch.Tensor, background: torch.Tensor, y: torch.Tensor):
        model.requires_grad_(False)
        self.noise_model.requires_grad_(True)
        torch.autograd.set_detect_anomaly(True)

        self.downsample_factor = [uniform(self.max_decrease_res,1) for batch in range(x.shape[0])]
        self._randomize_crop(y)
        self.rot_k = [choice([0,1,2,3]) for batch in range(x.shape[0])]
        self.rot_r = [uniform(-10,10) for batch in range(x.shape[0])]
    
        y = self._rand_rotate(y)
        y_crop = self._crop_sample(y)
        y_crop[y_crop<self.label_threshold]=0.
        y_crop[y_crop>=self.label_threshold]=1.

        adv_sample = self._create_adversarial_sample(x,background,y, False)

        loss_trajectory = []
        num_iters = 3
        for i in range(num_iters):
            with torch.cuda.amp.autocast():
                pred = model(adv_sample)
                loss: torch.Tensor = self.loss_fun(pred, y_crop)
                loss_trajectory.append(loss.item())
            self.scaler.scale(-loss).backward()
            if i == num_iters-1:
                self.noise_model.requires_grad_(False)
            adv_sample = self._create_adversarial_sample(x,background,y, True)
        model.requires_grad_(True)
        return adv_sample.detach(), y_crop

class DiceBCELoss():
    def __init__(self, sigmoid=False):
        super().__init__()
        if sigmoid:
            self.bce = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss(sigmoid=sigmoid)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return (self.dice(y_pred, y) + self.bce(y_pred, y))/2

class WeightedCosineLoss():
    def __init__(self, weights=[1,1,1]) -> None:
        self.weights = weights

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred_norm = torch.nn.functional.normalize(y_pred, dim=-1)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=y_pred.size(-1)).float()
        cosine_sim = torch.sum(y_pred_norm*y_one_hot, dim=1)
        sample_weights = torch.tensor([self.weights[y_i] for y_i in y], device=y.device)
        weighted_cosine_sim = sample_weights * cosine_sim
        return 1- (torch.sum(weighted_cosine_sim)/sample_weights.sum())


class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0, num_classes=3):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes

    def quadratic_kappa_coefficient(self, output, target):
        n_classes = target.shape[-1]
        weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
        weights = (weights - torch.unsqueeze(weights, -1)) ** 2

        C = (output.t() @ target).t()  # confusion matrix

        hist_true = torch.sum(target, dim=0).unsqueeze(-1)
        hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

        E = hist_true @ hist_pred.t()  # Outer product of histograms
        E = E / C.sum() # Normalize to the sum of C.

        num = weights * C
        den = weights * E

        QWK = 1 - torch.sum(num) / torch.sum(den)
        return QWK

    def quadratic_kappa_loss(self, output, target, scale=2.0):
        QWK = self.quadratic_kappa_coefficient(output, target)
        loss = -torch.log(torch.sigmoid(scale * QWK))
        return loss

    def forward(self, output, target):
        # Keep trace of output dtype for half precision training
        target = torch.nn.functional.one_hot(target.squeeze().long(), num_classes=self.num_classes).to(target.device, non_blocking=True).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return self.quadratic_kappa_loss(output, target, self.scale)

class WeightedMSELoss():
    def __init__(self, weights: list) -> None:
        self.weights = weights
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_per_sample = self.mse(y_pred, y)
        sample_weights = torch.tensor([self.weights[y_i] for y_i in y.long()], device=y.device)
        weighted_loss = loss_per_sample*sample_weights
        return torch.sum(weighted_loss)/sample_weights.sum()

class LSGANLoss(torch.nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0) -> None:
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = torch.nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss.mean()


def get_loss_function_by_name(name: str, config: dict[str, dict], scaler: GradScaler=None, loss=None) -> Union[DiceBCELoss, torch.nn.CrossEntropyLoss, WeightedCosineLoss]:
    if "Data" in config:
        weight = 1/torch.tensor(config["Data"]["class_balance"], device=config["General"].get("device") or "cpu")
    else:
        weight = None
    loss_map = {
        # "AtLoss": AtLoss(scaler, loss, None, 200/255, 1, alpha=1.25 * (100/255), grad_align_cos_lambda=0),
        "AtLoss": lambda: ANTLoss(scaler, loss, **(config[Phase.TRAIN].get("AT") or {})),
        "DiceBCELoss": lambda: DiceBCELoss(True),
        "CrossEntropyLoss": lambda: torch.nn.CrossEntropyLoss(weight=weight),
        "CosineEmbeddingLoss": lambda: WeightedCosineLoss(weights=weight),
        "MSELoss": lambda: torch.nn.MSELoss().to(device=config["General"].get("device") or "cpu", non_blocking=True),
        "WeightedMSELoss": lambda: WeightedMSELoss(weights=weight),
        "QWKLoss": lambda: QWKLoss(),
        "LSGANLoss": lambda: LSGANLoss().to(device=config["General"].get("device") or "cpu", non_blocking=True),
        "L1Loss": lambda: torch.nn.L1Loss().to(device=config["General"].get("device") or "cpu", non_blocking=True),
    }
    if name in loss_map:
        return loss_map[name]()
    else:
        print("Warning: No loss function defined. Ignore this message for parameterless models.")
        return lambda *args, **kwargs: None