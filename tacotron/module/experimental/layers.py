import torch
import torch.nn as nn
import torch.nn.functional as F
from tacotron.module.commons import fused_add_tanh_sigmoid_multiply


class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = (self.proj(x) * x_mask).transpose(1, 2)             # N x T x H
        m, logs = torch.split(stats, self.out_channels, dim=2)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask.transpose(1, 2)
        return z, m, logs


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                        dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask

###################### NANSY ###########################
class ConditionalLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, normalize_embedding: bool = True):
        super(ConditionalLayerNorm, self).__init__()
        self.normalize_embedding = normalize_embedding

        self.linear_scale = nn.Linear(embedding_dim, 1)
        self.linear_bias = nn.Linear(embedding_dim, 1)

    def forward(self, x, embedding):
        if self.normalize_embedding:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        scale = self.linear_scale(embedding).unsqueeze(-1)  # shape: (B, 1, 1)
        bias = self.linear_bias(embedding).unsqueeze(-1)  # shape: (B, 1, 1)

        out = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.var(x, dim=-1, keepdim=True)
        out = scale * out + bias
        return out


class ConvGLU(nn.Module):
    def __init__(self, channel, ks, dilation, embedding_dim=192, use_cLN=False):
        super(ConvGLU, self).__init__()

        self.dropout = nn.Dropout()
        self.conv = nn.Conv1d(channel, channel * 2, kernel_size=ks, stride=1, padding=(ks - 1) // 2 * dilation,
                              dilation=dilation)
        self.glu = nn.GLU(dim=1)  # channel-wise

        self.use_cLN = use_cLN
        if self.use_cLN:
            self.norm = ConditionalLayerNorm(embedding_dim)

    def forward(self, x, speaker_embedding=None):
        y = self.dropout(x)
        y = self.conv(y)
        y = self.glu(y)
        y = y + x

        if self.use_cLN and speaker_embedding is not None:
            y = self.norm(y, speaker_embedding)
        return y


class PreConv(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super(PreConv, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv1d(c_mid, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv1d(c_mid, c_out, kernel_size=1, dilation=1),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Generator(nn.Module):
    def __init__(self, c_in=1024, c_preconv=512, c_mid=128, c_out=80):
        super(Generator, self).__init__()

        self.network1 = nn.Sequential(
            PreConv(c_in, c_preconv, c_mid),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=9, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=27, use_cLN=False),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=9, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=27, use_cLN=False),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
        )

        self.LR = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(c_mid + 1, c_mid + 1, kernel_size=1, stride=1))

        self.network3 = nn.ModuleList([
            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=9, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=27, use_cLN=True),

            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=9, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=27, use_cLN=True),

            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
        ])

        self.lastConv = nn.Conv1d(c_mid + 1, c_out, kernel_size=1, dilation=1)

    def forward(self, x, energy, speaker_embedding):
        """
        Args:
            x: wav2vec feature or yingram. torch.Tensor of shape (B x C x t)
            energy: energy. torch.Tensor of shape (B x 1 x t)
            speaker_embedding: embedding. torch.Tensor of shape (B x d x 1)
        Returns:
        """
        y = self.network1(x)
        B, C, _ = y.shape

        y = F.interpolate(y, energy.shape[-1])  # B x C x d
        y = torch.cat((y, energy), dim=1)  # channel-wise concat
        y = self.LR(y)

        for module in self.network3:  # doing this since sequential takes only 1 input
            y = module(y, speaker_embedding)
        y = self.lastConv(y)
        return y


class Synthesis(nn.Module):
    def __init__(self):
        super(Synthesis, self).__init__()
        self.filter_generator = Generator(1024, 512, 128, 120)
        self.source_generator = Generator(50, 512, 128, 120)

    def forward(self, lps, s, e, ps):
        mel_filter = self.filter_generator(lps, e, s)
        mel_source = self.source_generator(ps, e, s)
        return mel_filter + mel_source


#####

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid=128, c_out=128):
        super(ResBlock, self).__init__()
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=(3 - 1) // 2 * 3, dilation=3)

        self.leaky_relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(c_mid, c_out, kernel_size=3, stride=1, padding=(3 - 1) // 2 * 3, dilation=3)

        self.conv3 = nn.Conv1d(c_in, c_out, kernel_size=1, dilation=1)

    def forward(self, x):
        y = self.conv1(self.leaky_relu1(x))
        y = self.conv2(self.leaky_relu2(y))
        y = y + self.conv3(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, conf=None):
        super(Discriminator, self).__init__()
        c_in = 80
        c_mid = 128
        c_out = 192

        self.phi = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=1, dilation=1),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
        )
        self.res = ResBlock(c_mid, c_mid, c_out)

        self.psi = nn.Conv1d(c_mid, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, mel, positive, negative):
        """
        Args:
            mel: mel spectrogram, torch.Tensor of shape (B x C x T)
            positive: positive speaker embedding, torch.Tensor of shape (B x d)
            negative: negative speaker embedding, torch.Tensor of shape (B x d)
        Returns:
        """
        pred1 = self.psi(self.phi(mel))
        pred = self.res(self.phi(mel))
        pred2 = torch.bmm(positive.unsqueeze(1), pred)
        pred3 = torch.bmm(negative.unsqueeze(1), pred)
        result = pred1 + pred2 - pred3
        result = result.squeeze(1)
        # result = torch.mean(result, dim=-1)
        return result
