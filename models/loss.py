import torch
import torch.nn as nn
import torch.nn.functional as F


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


# The larger the SI-SNR, the better the model
class SISNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        # return -torch.mean(si_snr(inputs, labels))
        return -(si_snr(x, y, eps=self.eps))


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        b, d, t = x.shape
        y[:, 0, :] = 0
        y[:, d // 2, :] = 0
        return F.mse_loss(x, y, reduction='mean') * d


class MAELoss(nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft

    def forward(self, x, y):
        gth_spec, gth_phase = self.stft(y)
        b, d, t = x.shape
        return torch.mean(torch.abs(x - gth_spec)) * d
