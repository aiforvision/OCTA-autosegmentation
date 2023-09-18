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
    def __init__(self, control_point_shape=(9,9), mode="bicubic") -> None:
        super().__init__()
        self.mode = mode
        self.control_point_shape = control_point_shape
        self.param_sampler = torch.distributions.beta.Beta(2,2)

    def reset_params(self, n_batch: int):
        if not hasattr(self, "alpha_unbound"):
            c_shape = (n_batch, 1, *self.control_point_shape)
            self.alpha_unbound = torch.nn.Parameter(torch.zeros(c_shape))
            self.beta_unbound = torch.nn.Parameter(torch.zeros(c_shape))
            self.register_parameter("alpha_unbound", self.alpha_unbound)
            self.register_parameter("beta_unbound", self.beta_unbound)
        rg = self.alpha_unbound.requires_grad
        self.requires_grad_(False)
        self.alpha_unbound[:,:,:,:] = 10**(self.param_sampler.sample(self.alpha_unbound.shape)*2-1)# torch.zeros_like(self.alpha_unbound).uniform_(-1,1)
        self.beta_unbound[:,:,:,:] =  10**(self.param_sampler.sample(self.beta_unbound.shape)*2-1)#torch.zeros_like(self.beta_unbound).uniform_(-1,1)
        self.requires_grad_(rg)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h, w = input.shape[-2:]
        Alpha_unbound = torch.nn.functional.interpolate(self.alpha_unbound, (h,w), mode="bicubic")
        Beta_unbound = torch.nn.functional.interpolate(self.beta_unbound, (h,w), mode="bicubic")

        A = torch.clamp(Alpha_unbound, min=1e-3)
        B = torch.clamp(Beta_unbound, min=1e-3)
        beta_dist = torch.distributions.beta.Beta(A,B)
        N: torch.Tensor = beta_dist.rsample()
        return N

class NoiseModel(torch.nn.Module):
    def __init__(self, grid_size=(9,9), lambda_delta = 1, lambda_speckle = 0.7, lambda_gamma=0.3, alpha=0.25) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.vessel_noise = ControlPointBetaNoise(self.grid_size)
        self.specle_noise = ControlPointBetaNoise(self.grid_size)
        self.lambda_delta = lambda_delta
        self.lambda_speckle = lambda_speckle
        self.lambda_gamma = lambda_gamma
        self.alpha = alpha
        self.optimizer = None

    def forward(self, I: torch.Tensor, I_d: torch.Tensor, adversarial: bool, downsample_factor=1) -> torch.Tensor:
        size = [s for s in I.shape[2:]]
        num_b = I.shape[0]
        I_new: torch.Tensor = torch.nn.functional.interpolate(I, scale_factor=1/downsample_factor, mode="bilinear")

        if self.optimizer is None:
            self.vessel_noise.reset_params(num_b)
            self.vessel_noise = self.vessel_noise.to(I.device)
            self.specle_noise.reset_params(num_b)
            self.specle_noise = self.specle_noise.to(I.device)
            self.control_points_gamma = torch.nn.Parameter(
                torch.zeros((num_b,1,*self.grid_size), dtype=I.dtype, device=I.device).uniform_(0,1)
            ).to(I.device)
            self.register_parameter("control_points_gamma", self.control_points_gamma)
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.alpha)
        if not adversarial:
            self.optimizer.zero_grad()
            self.vessel_noise.reset_params(num_b)
            self.specle_noise.reset_params(num_b)
            rg = self.control_points_gamma.requires_grad
            self.control_points_gamma.requires_grad_(False).uniform_(0,1).requires_grad_(rg)
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            Delta = self.vessel_noise.forward(I_new)[:num_b]
            N = self.specle_noise.forward(I_new)[:num_b]
            Gamma = torch.nn.functional.interpolate((torch.clamp(self.control_points_gamma,0,1)*(2*self.lambda_gamma)+(1-self.lambda_gamma))[:num_b], I_new.shape[-2:], mode="bicubic")

            D = I_d * self.lambda_delta * Delta
            I_new = torch.maximum(I_new, D)
            I_new = I_new * (self.lambda_speckle*N + (1-self.lambda_speckle))
            I_new = torch.pow(I_new+1e-6, Gamma)

            return torch.nn.functional.interpolate(I_new, size=size, mode="bilinear")