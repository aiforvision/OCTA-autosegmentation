from monai.losses import DiceLoss
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Union

from models.noise_model import NoiseModel

class AtLoss(torch.nn.Module):
    """
    Computes an adversarial training loss by finding a small perturbation r_adv via power iteration that when applied to the
    input image maximally increases the prediction error. The prediction model will then learn to generalize to this attack.
    """
    def __init__(self, scaler: GradScaler, loss_fun: torch.nn.Module, grid_size=None, eps=1.0, ip=1, alpha=1, grad_align_cos_lambda=0) -> None:
        """
        Parameters:
            - scaler: GradScaler for amp training
            - grad_size: control points for perturbation. For pixelwise attacks set to None
            - eps: magnitude of perturbation for r_adv
            - ip: Number of power iterations. If ip=1 then Fast gradient sign method (FGSM), else Projected Gradient Descent
            - alpha: step-size during power iteration
            - init: initialization type for noise vector. Either "zero" or "random"
            - grad_align_cos_lambda: factor for GradAlign regulizer to prevent catastrophic overfitting for FGSM
        """
        super(AtLoss, self).__init__()
        self.scaler = scaler
        self.loss_fun = loss_fun
        self.grid_size = grid_size
        self.eps = eps
        self.ip = ip
        self.alpha = alpha
        self.grad_align_cos_lambda = grad_align_cos_lambda

    def l2_norm_batch(self, v: torch.Tensor):
        """
        Computes the batchwise L2-norm for a 4D tensor
        """
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms

    def compute_grad_align(self, grad1: torch.Tensor, grad2: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient alignment regularization term as proposed by https://arxiv.org/abs/2007.02617.
        
        Parameters:
            - grad1: Gradient of r_adv when initialized with zero vector
            - grad2: Gradient of r_adv when initialized with random vector
        """
        grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
        grads_nnz_idx = (grad1_norms != 0) * (grad2_norms != 0)
        grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
        grad1_norms, grad2_norms = grad1_norms[grads_nnz_idx], grad2_norms[grads_nnz_idx]
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
        reg = (1.0 - cos.mean())
        return reg

    def compute_r_adv(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, r_init="zero", backprop=True):
        """
        Computes an adversarial noise vector via power iteration that when applied to the input image
        maximally increases the prediction error
        
        Parameters:
            - model: predictor model
            - x: input image
            - y: input label
            - r_init: initialization type for the noise vector
            - backprob: Whether to compute the gradient graph
        """
        grid_size = self.grid_size
        if grid_size is None:
            grid_size = x.shape
        r = torch.zeros(grid_size, device=x.device, dtype=x.dtype)
        if r_init == "uniform":
            r = r.uniform_(-self.eps, self.eps)
        r.requires_grad_()
        for i in range(self.ip):
            if r.grad is not None:
                r.grad.zero_()
            with torch.cuda.amp.autocast():
                pred = model(torch.clamp(x+r,0,1))
                loss: torch.Tensor = self.loss_fun(pred, y)
            self.scaler.scale(loss).backward()
            #r.grad.div_(self.scaler.get_scale())  # reverse back the scaling
                
            grad = r.grad.detach()
            r.data = r + self.alpha * self.eps * torch.sign(grad)
            r.data = torch.clamp(x+r.data, 0, 1) - x
            r.data = torch.clamp(r.data,-self.eps,self.eps)
        return r.detach(), grad
    
    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
        if self.grad_align_cos_lambda>0:
            # Additionally compute gradient aligment regularization term
            r, grad1 = self.compute_r_adv(model, x, y, "zero", False)
            _, grad2 = self.compute_r_adv(model, x, y, "uniform", True)
            reg = self.grad_align_cos_lambda * self.compute_grad_align(grad1,grad2)
        else:
            # Only compute adversarial noise vector
            r, _ = self.compute_r_adv(model, x, y, "uniform", True)
            reg = 0

        return x+r, reg

class ANTLoss(torch.nn.Module):
    def __init__(self, scaler: GradScaler, loss_fun: torch.nn.Module, grid_size=(9,9)) -> None:
        super().__init__()
        self.noise_model = NoiseModel(
            grid_size = grid_size,
            lambda_delta = 1,
            lambda_speckle = 0.7,
            lambda_gamma = 0.3,
            alpha=0.2
        )
        self.scaler = scaler
        self.loss_fun = loss_fun

    def forward(self, model: torch.nn.Module, x: torch.Tensor, background: torch.Tensor, y: torch.Tensor):
        model.requires_grad_(False)
        self.noise_model.requires_grad_(True)
        adv_sample = self.noise_model.forward(x, background, False, downsample_factor=4)
        loss_trajectory = []
        num_iters = 3
        for i in range(num_iters):
            with torch.cuda.amp.autocast():
                pred = model(adv_sample)
                loss: torch.Tensor = self.loss_fun(pred, y)
                loss_trajectory.append(loss.item())
            self.scaler.scale(-loss).backward()
            # with torch.cuda.amp.autocast():
            if i == i-num_iters:
                self.noise_model.requires_grad_(False)
            adv_sample = self.noise_model.forward(x, background, True, downsample_factor=4)
        model.requires_grad_(True)
        return adv_sample.detach(), 0

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
        target = torch.nn.functional.one_hot(target.squeeze().long(), num_classes=self.num_classes).to(target.device).type(output.dtype)
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


def get_loss_function_by_name(name: str, config: dict, scaler: GradScaler=None, loss=None) -> Union[DiceBCELoss, torch.nn.CrossEntropyLoss, WeightedCosineLoss]:
    if "Data" in config:
        weight = 1/torch.tensor(config["Data"]["class_balance"], device=config["General"].get("device") or "cpu")
    else:
        weight = None
    loss_map = {
        # "AtLoss": AtLoss(scaler, loss, None, 200/255, 1, alpha=1.25 * (100/255), grad_align_cos_lambda=0),
        "AtLoss": ANTLoss(scaler, loss, (9,9)),
        "DiceBCELoss": DiceBCELoss(True),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(weight=weight),
        "CosineEmbeddingLoss": WeightedCosineLoss(weights=weight),
        "MSELoss": torch.nn.MSELoss(),
        "WeightedMSELoss": WeightedMSELoss(weights=weight),
        "QWKLoss": QWKLoss(),
        "LSGANLoss": LSGANLoss().to(device=config["General"].get("device") or "cpu"),
    }
    return loss_map[name]