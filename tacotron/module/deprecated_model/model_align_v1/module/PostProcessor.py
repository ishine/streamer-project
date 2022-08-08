import torch.nn as nn
import torch.nn.utils.rnn as rnn
from tacotron.module.deprecated_model.model_align_v1.module.CBHG import CBHG


class PostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, num_filters):
        super(PostProcessor, self).__init__()
        self.CBHG = CBHG(dec_out_size, hidden_size, 2 * hidden_size, hidden_size, hidden_size, num_filters, True)
        self.projection = nn.Linear(2 * hidden_size, post_out_size)

    def forward(self, spec, **kwargs):
        lengths = kwargs.get('lengths')
        if lengths is None:
            N, T = spec.size(0), spec.size(1)
            lengths = [T for _ in range(N)]
            output = self.CBHG(spec, lengths).contiguous()
            output = self.projection(output)
        else:
            output = self.CBHG(spec, lengths)
            output = rnn.pack_padded_sequence(output, lengths, True, enforce_sorted=False)
            output = rnn.PackedSequence(self.projection(output.data), output.batch_sizes)
            output, _ = rnn.pad_packed_sequence(output, True)
        return output


class ConvPostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, dropout_p):
        super(ConvPostProcessor, self).__init__()
        self.net = nn.ModuleList([
            nn.Conv1d(dec_out_size, hidden_size, 5, padding=2),
            nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            ),
            nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            ),
            nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            ),
            nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, post_out_size, 5, padding=2),
            )
        ])

    def forward(self, spec, **kwargs):
        T_dec = spec.size(1)
        spec_mask = kwargs.get('spec_mask')
        if spec_mask is None:
            spec_mask = 1
        else:
            spec_mask = spec_mask[:, :, :T_dec]

        prev_output = spec.transpose(1,2) * spec_mask
        for i, m in enumerate(self.net):
            output = m(prev_output)
            if prev_output.size(1) == output.size(1):
                output = output + prev_output
            prev_output = output * spec_mask
        return prev_output.transpose(1,2)
