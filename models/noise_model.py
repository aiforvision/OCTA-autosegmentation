import torch
import random

def projected_gradient_ascent_step(prior: torch.Tensor, alpha=1, mode="PGA", lambda_x=1):
    if mode == "GS":
        return torch.clamp(torch.sign(prior.grad.detach()), 0, 1)
    elif mode == "PGA":
        return torch.clamp(prior.data + lambda_x*alpha*prior.grad.detach(),0,1)
    elif mode == "FGSM":
        return torch.clamp(prior.data + lambda_x*alpha*torch.sign(prior.grad.detach()),0,1)
    else:
        raise NotImplementedError()


class ControlPointBetaNoise(torch.nn.Module):
    
    def __init__(self, control_point_shape=(8,8), mode="bicubic", mean_interval = (0.05,0.95), alpha=0.25) -> None:
        super().__init__()
        self.control_point_shape = control_point_shape
        self.inference = False
        self.mode = mode
        self.dist = None
        self.mean_interval = mean_interval
        self.alpha = alpha
        self.seed = None

    def forward(self, input: torch.Tensor, adversarial: bool):
        if not adversarial:
            mean_control_points: torch.Tensor = torch.rand((*input.shape[:-2],*self.control_point_shape), device=input.device,dtype=input.dtype)*(self.mean_interval[1]-self.mean_interval[0])+self.mean_interval[0]
            mean_N = torch.nn.functional.interpolate(mean_control_points, input.shape[-2:], mode=self.mode)
            mean_N = torch.clamp(mean_N, *self.mean_interval)
            
            t = (mean_N*(1-mean_N))
            std_control_points: torch.Tensor = t.clone()-1e-6
            std_N = torch.nn.functional.interpolate(std_control_points, input.shape[-2:], mode=self.mode)
            self.seed = random.randint(0,1e6)
        else:
            posterior = projected_gradient_ascent_step(self.N, alpha=self.alpha, lambda_x=self.mean_interval[1]-self.mean_interval[0])
            mean_control_points = torch.nn.functional.interpolate(posterior, self.control_point_shape, mode=self.mode)
            mean_control_points = torch.clamp(mean_control_points, *self.mean_interval)
            mean_N = torch.nn.functional.interpolate(mean_control_points, input.shape[-2:], mode=self.mode)
            mean_N = torch.clamp(mean_N, *self.mean_interval)

            t = (mean_N*(1-mean_N))

            difference = (torch.clamp(posterior, *self.mean_interval) - mean_N)**2
            std_control_points = torch.sqrt(torch.clamp(torch.nn.functional.interpolate(difference, self.control_point_shape, mode=self.mode), 1e-6, 0.25))

            std_N = torch.nn.functional.interpolate(std_control_points, input.shape[-2:], mode=self.mode)
            std_N = torch.clamp(std_N, torch.tensor(0.01, device=t.device),t.sqrt()-1e-6)

        V = t / (std_N**2) - 1
        Alpha = mean_N * V
        Beta = (1-mean_N) * V

        Alpha = torch.nn.functional.relu(Alpha)+1e-6
        Beta = torch.nn.functional.relu(Beta)+1e-6

        torch.manual_seed(self.seed)

        self.dist = torch.distributions.beta.Beta(Alpha.float().cpu(),Beta.float().cpu())
        self.N: torch.Tensor = self.dist.sample().to(dtype=input.dtype, device=input.device)
        self.N.requires_grad_()
        return self.N

class NoiseModel(torch.nn.Module):
    def __init__(self, grid_size=(9,9), lambda_delta = 1, lambda_speckle = 0.5, lambda_gamma=0.01, alpha=0.25) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.vessel_noise = ControlPointBetaNoise(self.grid_size)
        self.specle_noise = ControlPointBetaNoise(self.grid_size)
        self.lambda_delta = lambda_delta
        self.lambda_speckle = lambda_speckle
        self.lambda_gamma = lambda_gamma
        self.alpha = alpha

    def forward(self, I: torch.Tensor, I_d: torch.Tensor, adversarial: bool, downsample_factor=4) -> torch.Tensor:
        size = [int(s*(4/downsample_factor)) for s in I.shape[2:]]
        I_new = torch.nn.functional.interpolate(I, scale_factor=1/downsample_factor, mode="bilinear")
        Delta = self.vessel_noise.forward(I_new, adversarial)
        N = self.specle_noise(I_new, adversarial)
        if not adversarial:
            self.control_points_gamma = torch.rand((*I.shape[:-2],*self.grid_size), dtype=I.dtype, device=I.device)*(2*self.lambda_gamma)+(1-self.lambda_gamma)
        else:
            # FGSM
            posterior = projected_gradient_ascent_step(self.control_points_gamma, alpha=self.alpha, lambda_x=2*self.lambda_gamma)
            self.control_points_gamma = torch.clamp(posterior, 1-self.lambda_gamma, 1+self.lambda_gamma)
        self.control_points_gamma.requires_grad_()

        with torch.cuda.amp.autocast():
            D = I_d * self.lambda_delta * Delta
            I_new = torch.maximum(I_new, D)

            I_new = I_new * (self.lambda_speckle*N + (1-self.lambda_speckle))

            
            Gamma = torch.nn.functional.interpolate(self.control_points_gamma, I_new.shape[-2:], mode="bicubic")
            I_new = torch.pow(I_new+1e-6, Gamma)
            return torch.nn.functional.interpolate(I_new, size=size, mode="bilinear")