import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np


class CBHG(nn.Module):
    """ input: NxTxinput_dim sized Tensor
        output: NxTx2gru_dim sized Tensor
    """
    def __init__(self, input_dim, conv_bank_dim, conv_dim1, conv_dim2, gru_dim, num_filters, is_masked):
        super(CBHG, self).__init__()
        self.num_filters = num_filters

        bank_out_dim = num_filters * conv_bank_dim
        self.conv_bank = nn.ModuleList()
        for i in range(num_filters):
            self.conv_bank.append(nn.Conv1d(input_dim, conv_bank_dim, i + 1, stride=1, padding=int(np.ceil(i / 2))))

        # define batch normalization layer, we use BN1D since the sequence length is not fixed
        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.InstanceNorm1d(bank_out_dim))
        self.bn_list.append(nn.InstanceNorm1d(conv_dim1))
        self.bn_list.append(nn.InstanceNorm1d(conv_dim2))

        self.conv1 = nn.Conv1d(bank_out_dim, conv_dim1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv_dim1, conv_dim2, 3, stride=1, padding=1)

        if input_dim != conv_dim2:
            self.residual_proj = nn.Linear(input_dim, conv_dim2)

        self.highway = Highway(conv_dim2, 4)
        self.rnn_residual = nn.Linear(conv_dim2, 2*conv_dim2)
        self.BGRU = nn.GRU(input_size=conv_dim2, hidden_size=gru_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input, lengths, spkr_vec_list=None):
        if spkr_vec_list is None:
            spkr_b1, spkr_b2, spkr_b3 = 0, 0, 0
        else:
            spkr_b1, spkr_b2, spkr_b3 = spkr_vec_list

        conv_bank_out = []
        for i in range(self.num_filters):
            tmp_input = input.transpose(1, 2)  # NxTxH -> NxHxT
            if i % 2 == 0:
                tmp_input = tmp_input.unsqueeze(-1)
                tmp_input = F.pad(tmp_input, (0,0,0,1)).squeeze(-1)   # NxHxT
            conv_bank_out.append(self.conv_bank[i](tmp_input) + spkr_b1)

        residual = torch.cat(conv_bank_out, dim=1)                  # NxHFxT
        residual = F.relu(self.apply_bn(residual, 0))
        residual = F.max_pool1d(residual, 2, stride=1)
        residual = self.conv1(residual)                             # NxHxT
        residual = F.relu(self.apply_bn(residual, 1)) + spkr_b2
        residual = self.conv2(residual)                             # NxHxT
        residual = self.apply_bn(residual, 2) + spkr_b3
        residual = residual.transpose(1,2)                          # NxHxT -> NxTxH

        rnn_input = input
        if rnn_input.size() != residual.size():
            rnn_input = self.residual_proj(rnn_input)
        rnn_input = rnn_input + residual
        rnn_input = self.highway(rnn_input)

        output = rnn.pack_padded_sequence(rnn_input, lengths.cpu(), True, enforce_sorted=False)
        output, _ = self.BGRU(output)                               # zero h_0 is used by default
        output, _ = rnn.pad_packed_sequence(output, True)           # NxTx2H

        rnn_residual = self.rnn_residual(rnn_input)
        output = rnn_residual + output
        return output

    def apply_bn(self, bn_input, bn_idx):
        if bn_input.size(2) < 2:
            return bn_input
        else:
            return self.bn_list[bn_idx](bn_input)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """ input: NxH sized Tensor
            output: NxH sized Tensor
        """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x
