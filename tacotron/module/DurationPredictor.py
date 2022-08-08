import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        spkr_embed_size=0,
        gst_size=0,
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
        self.gst_proj = nn.Linear(gst_size, filter_channels * 2)

    def forward(self, x, x_mask, spkr_vec=None, speed=None, gst=None, debug=0, **kwargs):
        N = x.size(0)
        spkr_biases = self.speaker_proj(spkr_vec).view(N, -1, 1)  # N x 2H x 1
        spkr_bias1, spkr_bias2 = torch.chunk(spkr_biases, 2, dim=1)
        speed_biases = self.speed_proj(speed.unsqueeze(1)).unsqueeze(2)
        speed_bias1, speed_bias2 = torch.chunk(speed_biases, 2, dim=1)
        if gst is None:
            gst_bias1 = gst_bias2 = 0
        else:
            gst_biases = self.gst_proj(gst).view(N, -1, 1)
            gst_bias1, gst_bias2 = torch.chunk(gst_biases, 2, dim=1)
        global_bias1 = spkr_bias1 + speed_bias1 + gst_bias1
        global_bias2 = spkr_bias2 + speed_bias2 + gst_bias2

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


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, hidden_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1, debug=0):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(
                input_size if i == 0 else hidden_size,
                hidden_size,
                kernel_size=kernel_size, 
                dropout=dropout,
                debug=debug
            )
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(hidden_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask, spkr_vec, debug=0):
        out = enc_out * enc_out_mask
        out, _ = self.layers((out.transpose(1, 2), spkr_vec))
        out = self.fc(out.transpose(1, 2)) * enc_out_mask
        return out


class TemporalPredictorRNN(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, hidden_size, spkr_embed_size,
                 n_layers=2, n_predictions=1, debug=0):
        super(TemporalPredictorRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size + spkr_embed_size,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(hidden_size, self.n_predictions, bias=True)

    def forward(self, enc_out, text_lengths, spkr_vec, debug=0):
        N, T_enc, _ = enc_out.size()
        output = rnn.pack_padded_sequence(
            torch.cat([enc_out, spkr_vec.expand(-1, T_enc, -1)], dim=2),
            text_lengths.cpu(),
            True,
            enforce_sorted=False
        )
        output, _ = self.lstm(output)
        output, _ = rnn.pad_packed_sequence(output, True)   # N x T x 2H
        output = torch.gather(output, 1, text_lengths.view(N, 1, 1).expand(-1, -1, output.size(2)) - 1)
        output = self.fc(output)            # N x 1 x 1
        return output


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, spkr_embed_size=64, debug=0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.spkr_bias = nn.Linear(spkr_embed_size, out_channels, bias=False)
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        signal, spkr_vec = inputs
        if spkr_vec is None:
            out = F.relu(self.conv(signal))
        else:
            out = F.relu(self.conv(signal) + self.spkr_bias(spkr_vec).transpose(1, 2))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out), spkr_vec
