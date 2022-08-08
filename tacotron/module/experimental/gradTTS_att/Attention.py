import torch, math
import torch.nn as nn
import torch.nn.functional as F

import tacotron.module.experimental.gradTTS_att.attentions as attentions
from tacotron.module.commons import maximum_path

from voxa.prep.signal_processing import SignalProcessing, dynamic_range_decompression_torch

class Attention(nn.Module):
    def __init__(self, enc_hidden, mel_dim, r_factor, debug=0, **kwargs):
        super(Attention, self).__init__()
        self.r_factor = r_factor

        n_vocab = kwargs.get('vocab_size')
        self.hidden_channels = hidden_channels = 192
        filter_channels = 768
        n_heads = 2
        n_layers = 6
        kernel_size = 3
        p_dropout = 0.1
        window_size = 4
        block_length = None
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        num_id = kwargs.get('num_id')
        spkr_hidden = 64
        self.spkr_emb = nn.Embedding(num_id, spkr_hidden)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            debug=debug,
            spkr_hidden=spkr_hidden
        )
            
        self.proj_output = nn.Linear(hidden_channels, 512)
        self.proj_m = nn.Conv1d(hidden_channels, mel_dim, 1)

    def forward(self, text, spec, text_lengths, spec_lengths, debug=0, **kwargs):
        """ spec: N x T x O_dec sized Tensor (Spectrogram)
            seq_style_vec: N x (T/r_factor) x H sized Tensor
        """
        N = spec.size(0)
        gst_vec = kwargs.get('gst_vec')
        spkr_vec = kwargs.get('spkr_vec').squeeze(1)

        # masking
        with torch.no_grad():
            text_mask = torch.arange(0, text.size(1), device=text.device).view(1, -1).expand(N, -1)
            text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, text.size(1)))                     # N x T_enc
            spec_lengths = spec_lengths // self.r_factor
            spec_mask = torch.arange(0, spec.size(1), device=text.device).view(1, -1).expand(N, -1)
            spec_mask = torch.lt(spec_mask, spec_lengths.view(-1, 1).expand(-1, spec.size(1)))                 # N x T_dec
            att_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(1)                                      # N x T_enc x T_dec

        text_mask = text_mask.unsqueeze(1)
        x = self.emb(text) * math.sqrt(self.hidden_channels) # [b, t, h]
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x = self.encoder(x, text_mask, spkr_vec, gst_vec=gst_vec)
        self.enc_output = self.proj_output(x.transpose(1, 2))
        x_m = self.proj_m(x) * text_mask
        text_mask = text_mask.squeeze(1)
        
        key = x_m.transpose(1, 2)                                   # N x T_enc x H_att
        query = spec

        loglikelihood = - 0.5 * torch.cdist(key, query).pow(2)      # N x T_enc x T_dec

        # hard att
        with torch.no_grad():
            attention = maximum_path(loglikelihood, att_mask).detach()

        # compute NLL
        nll_tile = -loglikelihood        
        att_loss = nll = torch.mean(
            torch.sum(nll_tile * attention, dim=(1, 2)) / spec_lengths
        )
        return attention, att_loss, att_mask, nll
