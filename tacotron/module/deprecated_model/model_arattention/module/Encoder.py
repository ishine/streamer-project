import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
import random 

import tacotron.module.guided_att_loss as guided_att


class Encoder(nn.Module):
    """ input: N x T
        spkr_vec: N x 1 x S
        return: N x T x H
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, dropout_p, spkr_embed_size, fluency=0, debug=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, charvec_dim)
        self.conv_1st = nn.Sequential(
            nn.Conv1d(charvec_dim, hidden_size, 5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 5, stride=1, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 5, stride=1, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )]
        )
        self.biLSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)

        if fluency == 0:
           self.spkr_fc = nn.Sequential(nn.Linear(spkr_embed_size, hidden_size * len(self.conv)), nn.Softsign())

    def forward(self, input, text_lengths, spkr_vec=None, debug=0):
        phoneme_emb = self.embedding(input)
        output = self.conv_1st(phoneme_emb.transpose(1, 2))                           # N x T x H -> N x H x T

        if spkr_vec is None:
            for i, layer in enumerate(self.conv):
                output = layer(output)                                                          # N x H x T
        else:
            spkr_bias = self.spkr_fc(spkr_vec).transpose(1, 2)                                  # N x 2H x 1
            spkr_bias = torch.split(spkr_bias, spkr_bias.size(1) // len(self.conv), dim=1)

            for i, layer in enumerate(self.conv):
                output = layer(output) + spkr_bias[i]                                           # N x H x T

        if text_lengths is None:
            output, _ = self.biLSTM(output.transpose(1, 2))                                     # N x H x T -> N x T x H
        else:
            output = rnn.pack_padded_sequence(output.transpose(1, 2), text_lengths, True, enforce_sorted=False)
            output, _ = self.biLSTM(output)
            output, _ = rnn.pad_packed_sequence(output, True)                                   # NxTx2H
        
        return phoneme_emb, output


class TsideAtt(nn.Module):
    """ text_encoding: N x T_enc x H
        key_vec: N x T_dec x H
    """
    def __init__(self, query_dim, key_dim, hidden_size, r_factor, **kwargs):
        super(TsideAtt, self).__init__()
        self.exp_no = kwargs.get('exp_no')
        self.r_factor = r_factor
        self.query_proj = nn.Linear(query_dim, hidden_size, bias=False)
        self.key_proj = nn.Linear(key_dim, hidden_size, bias=False)        
        self.mask_maker = guided_att.GuidedAttentionLoss()

    def forward(self, query_vec, key_vec, value_vec, text_lengths, spec_lengths):
        projed_query = self.query_proj(query_vec)  # N x T_enc x H
        projed_key = self.key_proj(key_vec).transpose(1, 2)  # N x H x T_dec

        # stable softmax
        logit = torch.bmm(projed_query, projed_key)         # N x T_enc x T_dec
        logit_max, _ = torch.max(logit, dim=2, keepdim=True)
        att_weights = torch.exp(logit - logit_max)

        if hasattr(self, 'mask_maker'):
            spec_lengths = (spec_lengths // self.r_factor).long()
            att_mask = self.mask_maker._make_masks(text_lengths, spec_lengths).transpose(1, 2)
            att_weights = att_weights * att_mask.type_as(spec_lengths)

        att_weights = F.normalize(att_weights, 1, 2)        # N x T_enc x T_dec
        value_vec = torch.bmm(att_weights, value_vec)       # N x T_enc x P
        return value_vec, att_weights


class ProsodyPredictorWrapper(nn.Module):
    """ input: N x T
        spkr_vec: N x 1 x S
        return: N x T x H
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, dropout_p, spkr_embed_size, seq_prosody_dim,
                 embed_hidden, key_dim, debug=0, **kwargs):
        super(ProsodyPredictorWrapper, self).__init__()
        self.exp_no = kwargs.get('exp_no')
        self.encoder = Encoder(vocab_size, charvec_dim, hidden_size, dropout_p, spkr_embed_size, debug=debug, fluency=0)
        self.speed_proj = nn.Sequential(
            nn.Linear(1, embed_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(embed_hidden, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p)
        )

        self.prosody_predictor = ProsodyPredictor(embed_hidden, hidden_size, seq_prosody_dim, self.exp_no)

    def forward(self, input, text_lengths, spkr_vec, style_vec=None, seq_prosody_vec=None, speed=None, debug=0, **kwargs):
        _, text_encoding = self.encoder(input, text_lengths, spkr_vec, debug)
        text_encoding = text_encoding + style_vec + self.speed_proj(speed.view(-1, 1, 1))

        style_vec_prediction = self.prosody_predictor(text_encoding, text_lengths, seq_prosody_vec)
        return text_encoding, style_vec_prediction


class ProsodyPredictor(nn.Module):
    def __init__(self, prosody_predictor_dim, text_hidden, seq_prosody_hidden, exp_no):
        super(ProsodyPredictor, self).__init__()
        self.exp_no = exp_no
        self.seq_prosody_hidden = seq_prosody_hidden
        self.prosody_predictor = nn.GRU(text_hidden + seq_prosody_hidden, prosody_predictor_dim, batch_first=True)
        self.prosody_predictor_proj = nn.Linear(prosody_predictor_dim, seq_prosody_hidden)

    def forward(self, text_encoding, text_lengths, seq_prosody_vec=None):
        N, T_enc, _ = text_encoding.size()

        # predict text-side sequential prosody to use in inference
        padding = text_encoding.new().resize_(N, 1, self.seq_prosody_hidden).zero_()
        if self.training:
            seq_prosody_vec = seq_prosody_vec.detach()
            seq_prosody_vec_shifted = torch.cat([padding, seq_prosody_vec[:, :-1]], dim=1)
            autoregressive_input = torch.cat([text_encoding, seq_prosody_vec_shifted], dim=2)   # N x T_enc x (H+P)
            autoregressive_input = rnn.pack_padded_sequence(autoregressive_input, text_lengths, True, enforce_sorted=False)

            style_vec_prediction, _ = self.prosody_predictor(autoregressive_input)              # N x T_enc x H
            style_vec_prediction, _ = rnn.pad_packed_sequence(style_vec_prediction, True)
            style_vec_prediction = self.prosody_predictor_proj(style_vec_prediction)
        else:
            pred_list = []
            prev_prosody_vec = padding
            gru_hidden = None
            for i in range(max(text_lengths)):
                autoregressive_input = torch.cat([text_encoding[:, i:i + 1], prev_prosody_vec], dim=2)
                if gru_hidden is None:
                    prev_prosody_vec, gru_hidden = self.prosody_predictor(autoregressive_input)
                else:
                    prev_prosody_vec, gru_hidden = self.prosody_predictor(autoregressive_input, gru_hidden)    
                prev_prosody_vec = self.prosody_predictor_proj(prev_prosody_vec)
                pred_list.append(prev_prosody_vec)
            style_vec_prediction = torch.cat(pred_list, dim=1)
        return style_vec_prediction


class ProsodyPredictorNONAR(nn.Module):
    def __init__(self, prosody_predictor_dim, text_hidden, seq_prosody_hidden, exp_no):
        super(ProsodyPredictorNONAR, self).__init__()
        self.exp_no = exp_no
        self.seq_prosody_hidden = seq_prosody_hidden
        self.prosody_predictor = nn.GRU(text_hidden, prosody_predictor_dim, batch_first=True)
        self.prosody_predictor_proj = nn.Linear(prosody_predictor_dim, seq_prosody_hidden)
    
    def forward(self, text_encoding, text_lengths, seq_prosody_vec=None):
        N, T_enc, _ = text_encoding.size()

        text_encoding = text_encoding.detach()
        autoregressive_input = rnn.pack_padded_sequence(text_encoding, text_lengths, True, enforce_sorted=False)
        style_vec_prediction, _ = self.prosody_predictor(autoregressive_input)              # N x T_enc x H
        style_vec_prediction, _ = rnn.pad_packed_sequence(style_vec_prediction, True)

        style_vec_prediction = self.prosody_predictor_proj(style_vec_prediction)
        return style_vec_prediction


class EncoderConv(nn.Module):
    """ input: N x T
        spkr_vec: N x 1 x S
        return: N x T x H
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, spkr_embed_size, debug=0):
        super(EncoderConv, self).__init__()
        assert charvec_dim % 2 == 0

        num_layer = 7
        charvec_dim = hidden_size
        self.gate_fc = nn.Linear(hidden_size, 1)

        self.embedding = nn.Embedding(vocab_size, charvec_dim, scale_grad_by_freq=True)

        in_channel = charvec_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(ResBlock(in_channel, hidden_size, 5, spkr_size=spkr_embed_size))
            in_channel = hidden_size

        self.key_fc = nn.Linear(hidden_size, charvec_dim)

    def forward(self, input, text_lengths, spkr_vec=None, debug=0):
        char_embedding = self.embedding(input)                                  # N x T x H

        output = char_embedding.transpose(1, 2)                             # N x T x H -> N x H x T

        for i, block in enumerate(self.layers):
            output = block(output, spkr_vec=spkr_vec, debug=debug)              # N x H x T

        gate = torch.sigmoid(self.gate_fc(output.transpose(1, 2)))              # N x T x (H or 1)

        output = gate * char_embedding + (1-gate) * output.transpose(1, 2)
        return output


class ResBlock(nn.Module):
    """ input: N x H x T
        spkr_vec: N x 1 x S
        return: N x H x T
    """
    def __init__(self, in_channels, out_channels, kernel_size, spkr_size=0, dropout=0.05, debug=0):
        super(ResBlock, self).__init__()
        self.conv_channels = out_channels
        self.dropout = dropout

        padding = 1 * (kernel_size - 1) // 2        # we multiply 1 since we are assuming dilation 1,
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels, 2 * out_channels, kernel_size, padding=padding))

        self.spkr_fc = nn.Linear(spkr_size, out_channels)

    def forward(self, x, spkr_vec=None, debug=0):
        res = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        conv_out = self.conv(x)

        h_filter = torch.tanh(conv_out[:, :self.conv_channels])
        h_gate = torch.sigmoid(conv_out[:, self.conv_channels:])
        if not spkr_vec is None:
            spkr_bias = self.spkr_fc(spkr_vec.squeeze(1)).unsqueeze(-1)
            h_filter = h_filter + spkr_bias
        output = h_filter * h_gate

        output = (output + res) * np.sqrt(0.5)
        return output
