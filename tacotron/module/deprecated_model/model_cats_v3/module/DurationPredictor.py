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


class TemporalPredictorRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, dec_hidden, output_size, spkr_embed_size, dropout_p=0.1, debug=0, ar_input_size=0):
        super(TemporalPredictorRNN, self).__init__()
        self.num_lstm_layers = 2

        if ar_input_size > 0:
            prenet_in_size = ar_input_size
        else:
            prenet_in_size = output_size
        self.prenet = nn.Linear(prenet_in_size, dec_hidden)

        in_lstm_size = enc_hidden + dec_hidden + spkr_embed_size
        self.LSTM = nn.LSTM(
            in_lstm_size,
            dec_hidden,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )

        self.out_linear = nn.Linear(enc_hidden + dec_hidden, output_size)
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, debug=0, **kwargs):
        N = input_enc.size(0)
        if self.null_state:
            self.null_state = False

        out_prenet = self.prenet(input_dec)  # N x T_dec x H
        
        spkr_vec_expanded = spkr_vec.expand(-1, out_prenet.size(1), -1)
        in_lstm = torch.cat((out_prenet, input_enc, spkr_vec_expanded), 2)        # N x T_dec x 4H + 1

        if self.hidden is None:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, None)           # N x T_dec x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, (self.hidden, self.cell))  # N x T_dec x 4H, L x N x 4H, L x N x 4H

        dec_output = torch.cat((out_lstm, input_enc), 2)                              # N x T_dec x 6H
        output = self.out_linear(dec_output)
        return output

    def reset_states(self, debug=0):
        # need to reset states at every sub-batch (to consider TBPTT)
        self.hidden = None
        self.cell = None
        self.null_state = True
