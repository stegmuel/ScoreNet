"""
Differentiable top-n_patches using perturbed input for gradient computation.
See https://q-berthet.github.io/papers/BerBloTeb20.pdf.
Source: https://arxiv.org/pdf/2104.03059.pdf
"""
from torch.nn.functional import one_hot
import torch.nn as nn
import torch
import math


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05, sigma_decay: float = 0.9):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.noise_is_zero = False
        self.k = k

    def update_sigma(self):
        """
        Decreases the value of sigma such that the effective number of selected patches is close to the desired one.
        :return: None.
        """
        if not self.noise_is_zero:
            self.sigma *= self.sigma_decay
        if math.isclose(self.sigma, 0., abs_tol=1e-6):
            self.noise_is_zero = True

    def __call__(self, x, k):
        return PerturbedTopKFunction.apply(x, k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        # b, nS, k, d
        perturbed_output = one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 4)
        noise_gradient = ctx.noise

        # Sum over the estimation i.e. take the expectation
        expected_gradient = (
                torch.einsum('b n k d, b n d -> b k d', ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / ctx.sigma
        )
        grad_input = torch.einsum('b k d, b k d -> b d', grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 3)
