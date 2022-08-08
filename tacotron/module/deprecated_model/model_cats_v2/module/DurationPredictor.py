import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        spkr_embed_size=-1,
        debug=0,
        **kwargs
    ):
        super(DurationPredictor, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # speaker/speed biases will be sliced.
        self.speaker_proj = nn.Linear(spkr_embed_size, filter_channels * 2)
        self.speed_proj = nn.Linear(1, filter_channels * 2)

    def forward(self, x, x_mask, spkr_vec=None, speed=None, debug=0, **kwargs):
        N = x.size(0)
        spkr_biases = self.speaker_proj(spkr_vec).view(N, -1, 1)  # N x 2H x 1
        speed_biases = self.speed_proj(speed.unsqueeze(1)).unsqueeze(2)
        spkr_bias1, spkr_bias2 = torch.chunk(spkr_biases, 2, dim=1)
        speed_bias1, speed_bias2 = torch.chunk(speed_biases, 2, dim=1)
        global_bias1 = spkr_bias1 + speed_bias1
        global_bias2 = spkr_bias2 + speed_bias2

        x = self.conv_1(x * x_mask) + global_bias1
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask) + global_bias2
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return (x * x_mask).squeeze(1)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
