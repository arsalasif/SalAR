import torch
import torch.nn as nn
import numpy as np


class GaussianPrior(nn.Module):
    def __init__(self, input_channels, nb_gaussian):
        super(GaussianPrior, self).__init__()
        self.input_channels = input_channels
        self.nb_gaussian = nb_gaussian

        w = (self.nb_gaussian*4, )
        self.weight = nn.Parameter(self.gaussian_priors_init(w))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ep = 1e-07
        mu_x = self.weight.data[0:self.nb_gaussian]
        mu_y = self.weight.data[self.nb_gaussian:self.nb_gaussian*2]
        sigma_x = self.weight.data[self.nb_gaussian*2:self.nb_gaussian*3]
        sigma_y = self.weight.data[self.nb_gaussian*3:]

        b_s = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        e = height / width
        e1 = (1 - e) / 2
        e2 = e1 + e

        mu_x = torch.clamp(mu_x, min=0.25, max=0.75)
        mu_y = torch.clamp(mu_y, min=0.35, max=0.65)

        sigma_x = torch.clamp(sigma_x, min=0.1, max=0.9)
        sigma_y = torch.clamp(sigma_y, min=0.2, max=0.8)

        # check unsqueeze

        x_t = torch.mm(torch.ones((height, 1)), torch.linspace(0, 1.0, width).unsqueeze(0)).double().to(torch.device(device=x.get_device()))
        y_t = torch.mm(torch.linspace(e1, e2, height).unsqueeze(-1), torch.ones((1, width))).double().to(torch.device(device=x.get_device()))

        # check repeat
        x_t = torch.unsqueeze(x_t, -1).repeat_interleave(self.nb_gaussian, dim=-1)
        y_t = torch.unsqueeze(y_t, -1).repeat_interleave(self.nb_gaussian, dim=-1)

        exp = torch.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + ep) + (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + ep)))
        gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + ep) * exp


        gaussian = gaussian.permute(2, 0, 1)

        max_gauss = torch.repeat_interleave(torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.max(torch.max(gaussian, dim=1)[0], dim=1)[0], dim=-1), height, dim=-1), -1), width, dim=-1)

        gaussian = gaussian / max_gauss

        output = torch.repeat_interleave(torch.unsqueeze(gaussian, dim=0), b_s, dim=0).float().to(torch.device(device=x.get_device()))
        output = self.relu(output)
        return output

    # Gaussian priors initialization
    def gaussian_priors_init(self, shape, name=None):
        means = np.random.uniform(low=0.3, high=0.7, size=shape[0] // 2)
        covars = np.random.uniform(low=0.05, high=0.3, size=shape[0] // 2)
        return torch.from_numpy(np.concatenate((means, covars), axis=0))
