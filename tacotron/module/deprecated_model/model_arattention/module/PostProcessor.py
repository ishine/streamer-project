import torch.nn as nn
import torch.nn.utils.rnn as rnn
from tacotron.module.deprecated_model.model_arattention.module.CBHG import CBHG


class PostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, num_filters):
        super(PostProcessor, self).__init__()
        self.CBHG = CBHG(dec_out_size, hidden_size, 2 * hidden_size, hidden_size, hidden_size, num_filters, True)
        self.projection = nn.Linear(2 * hidden_size, post_out_size)

    def forward(self, input, lengths=None):
        if lengths is None:
            N, T = input.size(0), input.size(1)
            lengths = [T for _ in range(N)]
            output = self.CBHG(input, lengths).contiguous()
            output = self.projection(output)
        else:
            output = self.CBHG(input, lengths)
            output = rnn.pack_padded_sequence(output, lengths, True, enforce_sorted=False)
            output = rnn.PackedSequence(self.projection(output.data), output.batch_sizes)
            output, _ = rnn.pad_packed_sequence(output, True)
        return output


class NewPostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, dropout_p):
        super(NewPostProcessor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dec_out_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, post_out_size, 5, padding=2),
        )

    def forward(self, input, lengths=None):
        return self.net(input.transpose(1,2)).transpose(1,2)
