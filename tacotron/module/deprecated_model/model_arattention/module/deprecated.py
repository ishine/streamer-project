# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import gammaln


class AttnDecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, att_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, r_factor=2, dropout_p=0.5, debug=0):
        super(AttnDecoderRNN, self).__init__()
        self.r_factor = r_factor
        self.H_dec = dec_hidden
        self.O_dec = output_size
        self.num_lstm_layers = 2

        def bias_layer(in_dim, out_dim, bias=True):
            return nn.Sequential(
                        nn.Linear(in_dim, out_dim, bias=bias),
                        nn.Softsign()
                   )

        # outputs of the following layers are reusable through recurrence
        self.in_att_linear_enc = bias_layer(enc_hidden, att_hidden, bias=True)
        self.in_att_linear_spkr = bias_layer(spkr_embed_size, att_hidden, bias=False)
        self.in_att_conv_prev_att = nn.Conv1d(1, att_hidden, 31, padding=15, bias=False)

        self.in_att_linear_dec = nn.Linear(2 * (4 * dec_hidden), att_hidden, bias=False)
        self.att_proj = nn.Linear(att_hidden, 1, )

        self.prenet = nn.Sequential(
            nn.Linear(output_size + spkr_embed_size, 2 * dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.LSTM = nn.LSTM(enc_hidden + dec_hidden + spkr_embed_size, 4 * dec_hidden,
                            num_layers=self.num_lstm_layers, batch_first=True)

        self.out_linear = nn.Linear(enc_hidden + 4 * dec_hidden, output_size * r_factor)
        self.set_attention_range(att_range)
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, lengths_enc, debug=0, context_vec=None):
        N, T_enc = input_enc.size(0), max(lengths_enc)

        if self.null_state:
            a0 = input_enc.data.new().resize_(N, T_enc, 1).zero_()
            a0[:, 0].fill_(1)                                                               # force initial attention
            self.att_weights = a0.data
            self.null_state = False

        if self.null_bias:
            # reusable bias terms
            self.att_bias_enc = self.in_att_linear_enc(input_enc)                           # N x T_enc x H_att
            self.att_bias_spkr = self.in_att_linear_spkr(spkr_vec).expand_as(self.att_bias_enc)
            self.null_bias = False

        if input_dec is None:
            input_dec = input_enc.data.new().resize_(N, self.O_dec).zero_()

        input_dec = torch.cat([input_dec, spkr_vec.squeeze(1)], dim=-1)
        out_prenet = self.prenet(input_dec).unsqueeze(1)  # N x O_dec -> N x 1 x H

        # attention -- https://arxiv.org/pdf/1506.07503.pdf
        self.att_weights = self.att_weights[:, :max(lengths_enc)]
        in_att_prev_att = self.in_att_conv_prev_att(self.att_weights.transpose(1, 2)).transpose(1, 2)

        if self.hidden is None:
            in_att_dec = 0
        else:
            in_att_dec = self.in_att_linear_dec(self.hidden.transpose(0, 1).contiguous().view(N, 1, -1))

        e = self.att_bias_enc + in_att_dec + self.att_bias_spkr + in_att_prev_att  # N x T_enc x H_att

        # attention mask (confine attention to be formed near previously attended characters)
        with torch.no_grad():
            att_mask = self.att_weights.data.new().resize_(N, T_enc).zero_()
            _, att_max_idx = torch.max(self.att_weights.data, dim=1)
            for i in range(self.att_range):
                idx1 = torch.min(torch.clamp((att_max_idx + i), min=0), torch.Tensor(lengths_enc).sub(1).type_as(att_max_idx)).long()
                idx2 = torch.min(torch.clamp((att_max_idx - i), min=0), torch.Tensor(lengths_enc).sub(1).type_as(att_max_idx)).long()
                att_mask.scatter_(1, idx1, 1)
                att_mask.scatter_(1, idx2, 1)
            att_mask = att_mask.view(N, T_enc, 1)

        # stable softmax
        logit = self.att_proj(torch.tanh(e))
        logit_max, _ = torch.max(logit, dim=1, keepdim=True)
        self.att_weights = torch.exp(logit - logit_max) * att_mask
        self.att_weights = F.normalize(self.att_weights, 1, 1)                      # N x T_enc x 1

        context = torch.bmm(self.att_weights.transpose(1, 2), input_enc)  # N x 1 x 2H

        in_lstm = torch.cat((out_prenet, context, spkr_vec), 2)        # N x 1 x 4H

        if self.hidden is None:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, None)           # N x 1 x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, (self.hidden, self.cell))  # N x 1 x 4H, L x N x 4H, L x N x 4H

        dec_output = torch.cat((out_lstm, context), 2)                      # N x 1 x 6H
        output = self.out_linear(dec_output).view(N, self.r_factor, -1)     # N x r x O_dec
        return {
            "output_dec": output,
            "context": context,
            "output_lstm": dec_output,
        }

    def set_attention_range(self, range):
        self.att_range = range

    def reset_states(self, debug=0):
        # need to reset states at every sub-batch (to consider TBPTT)
        self.hidden = None
        self.cell = None
        self.att_weights = None

        self.prev_kappa = 0
        self.prev_ctx = None

        self.null_state = True

    def reset_bias(self):
        # need to reset bias at every iteration to avoid unnecessary computation
        self.att_bias_enc = None
        self.att_bias_spkr = None

        self.null_bias = True

        self.att_bias_style = None

    def mask_states(self, len_mask, debug=0):
        if not self.null_state:
            if len_mask is None:
                self.hidden = self.hidden.data
                self.cell = self.cell.data
                self.att_weights = self.att_weights.data
            else:
                self.hidden  = torch.index_select(self.hidden.data, 1, len_mask).data
                self.cell  = torch.index_select(self.cell.data, 1, len_mask).data
                self.att_weights  = torch.index_select(self.att_weights.data, 0, len_mask).data

    def get_velocity_loss(self):
        loss = self.velocity_loss
        self.velocity_loss = 0
        return loss


class LRAttnDecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, att_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, r_factor=2, dropout_p=0.5, debug=0, num_t_layer=2):
        super(LRAttnDecoderRNN, self).__init__()
        self.r_factor = r_factor
        self.H_enc = enc_hidden
        self.H_att = att_hidden
        self.H_dec = dec_hidden
        self.O_dec = output_size
        self.num_lstm_layers = 2
        self.dca_filter_size = 21
        self.prior_filter_size = 11

        # outputs of the following layers are reusable through recurrence
        self.in_att_linear_spkr = nn.Sequential(
            nn.Linear(spkr_embed_size, att_hidden, bias=False),
            nn.Softsign()
        )
        self.in_att_conv_prev_att = nn.Conv1d(1, att_hidden // 16, 31, padding=15)
        self.in_att_proj_prev_att = nn.Linear(att_hidden // 16, att_hidden, bias=False)

        self.register_buffer('prior_filter', torch.zeros(1, 1, self.prior_filter_size))
        a = 0.1
        b = 0.9
        n = 10
        for x in range(self.prior_filter_size):
            self.prior_filter[0, 0, x] = gammaln(n+1) + gammaln(x+a) + gammaln(n-x+b) + gammaln(a+b) - \
                                        (gammaln(x+1) + gammaln(n-x+1) + gammaln(a) + gammaln(b) + gammaln(n+a+b))
        self.prior_filter = torch.exp(self.prior_filter)

        self.att_proj = nn.Linear(att_hidden, 1, )

        self.in_att_speed = nn.Linear(1, att_hidden, bias=False)
        self.speed_proj = nn.Sequential(
            nn.Linear(1, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dec_hidden, enc_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_p)
        )

        self.prenet = nn.Sequential(
            nn.Linear(output_size + spkr_embed_size, 2 * dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.attRNN = nn.GRU(enc_hidden + dec_hidden, att_hidden, batch_first=True)

        self.dca_linear = nn.Sequential(
            nn.Linear(att_hidden, att_hidden),
            nn.Tanh(),
            nn.Linear(att_hidden, (att_hidden // 16) * self.dca_filter_size, bias=False),
        )
        self.dca_conv = BatchConv1DLayer(1, att_hidden // 16, padding=int(self.dca_filter_size // 2))
        self.dca_proj = nn.Linear(att_hidden // 16, att_hidden, bias=False)

        self.LSTM = nn.LSTM(enc_hidden + dec_hidden + spkr_embed_size, 4 * dec_hidden,
                            num_layers=self.num_lstm_layers, batch_first=True)

        # self.out_linear = nn.Linear(enc_hidden + 4 * dec_hidden, output_size * r_factor)
        if num_t_layer == 4:
            trans_fc_hiddden = dec_hidden // 2
        else:
            trans_fc_hiddden = dec_hidden

        self.out_linear = nn.Sequential(
            nn.Linear(enc_hidden + dec_hidden + spkr_embed_size + 4 * dec_hidden, dec_hidden),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(dec_hidden, 8, 2 * trans_fc_hiddden), num_t_layer),
            nn.Linear(dec_hidden, output_size * r_factor),
        )

        self.set_attention_range(att_range)
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, lengths_enc, speed, debug=0, context_vec=None):
        N, T_enc = input_enc.size(0), max(lengths_enc)
        in_att_speed = self.in_att_speed(speed.unsqueeze(-1)).unsqueeze(1)

        if self.null_state:
            a0 = input_enc.data.new().resize_(N, T_enc, 1).zero_()
            a0[:, 0].fill_(1)                                                               # force initial attention
            self.att_weights = a0.data
            self.context = input_enc.data.new().resize_(N, 1, self.H_enc).zero_()
            self.null_state = False

        if self.null_bias:
            # reusable bias terms
            self.att_bias_spkr = self.in_att_linear_spkr(spkr_vec).expand(-1, T_enc, -1)
            self.null_bias = False

        if input_dec is None:
            input_dec = input_enc.data.new().resize_(N, self.O_dec).zero_()
        input_dec = torch.cat([input_dec, spkr_vec.squeeze(1)], dim=-1)
        out_prenet = self.prenet(input_dec).unsqueeze(1)  # N x O_dec -> N x 1 x H

        # location-relative attention -- https://arxiv.org/abs/1910.10288
        att_weight_T = self.att_weights.transpose(1, 2)
        in_att_prev_att = self.in_att_proj_prev_att(self.in_att_conv_prev_att(att_weight_T).transpose(1, 2))

        in_att_rnn = torch.cat([out_prenet, self.context], 2)
        out_att_rnn, self.att_hidden = self.attRNN(in_att_rnn, self.att_hidden)
        dca_filter = self.dca_linear(out_att_rnn).view(N, self.H_att // 16, 1, self.dca_filter_size).contiguous()
        dca_x = att_weight_T.unsqueeze(1)
        in_att_curr_att_rnn = self.dca_proj(self.dca_conv(dca_x, dca_filter).squeeze(1).transpose(1, 2))

        e = in_att_prev_att + in_att_curr_att_rnn + self.att_bias_spkr + in_att_speed     # N x T_enc x H_att

        # attention mask (confine attention to be formed near previously attended characters)
        with torch.no_grad():
            att_mask = self.att_weights.data.new().resize_(N, T_enc).zero_()
            _, att_max_idx = torch.max(self.att_weights.data, dim=1)
            for i in range(self.att_range):
                idx1 = torch.min(torch.clamp((att_max_idx + i), min=0), torch.Tensor(lengths_enc).sub(1).type_as(att_max_idx)).long()
                idx2 = torch.min(torch.clamp((att_max_idx - i), min=0), torch.Tensor(lengths_enc).sub(1).type_as(att_max_idx)).long()
                att_mask.scatter_(1, idx1, 1)
                att_mask.scatter_(1, idx2, 1)
            att_mask = att_mask.view(N, T_enc, 1)
        # att_mask = 1

        # prior filter
        in_prior = F.conv1d(att_weight_T, self.prior_filter, padding=int(self.prior_filter_size // 2)).transpose(1, 2)
        logit_prior_bias = torch.log(torch.clamp(in_prior, min=1e-6))

        # stable softmax
        logit = self.att_proj(torch.tanh(e)) + logit_prior_bias
        logit_max, _ = torch.max(logit, dim=1, keepdim=True)
        self.att_weights = torch.exp(logit - logit_max) * att_mask
        self.att_weights = F.normalize(self.att_weights, 1, 1)                      # N x T_enc x 1

        input_enc = input_enc + self.speed_proj(speed.unsqueeze(-1)).unsqueeze(1)

        context = torch.bmm(self.att_weights.transpose(1, 2), input_enc)  # N x 1 x 2H

        in_lstm = torch.cat((out_prenet, context, spkr_vec), 2)        # N x 1 x 4H

        if self.hidden is None:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, None)           # N x 1 x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, (self.hidden, self.cell))  # N x 1 x 4H, L x N x 4H, L x N x 4H

        dec_output = torch.cat((out_lstm, in_lstm), 2).transpose(0, 1)                              # N x 1 x 6H

        output = self.out_linear(dec_output).transpose(0, 1).view(N, self.r_factor, -1)                 # N x r x O_dec
        return {
            "output_dec": output,
            "context": context,
            "output_lstm": dec_output,
        }

    def set_attention_range(self, range):
        self.att_range = range

    def reset_states(self, debug=0):
        # need to reset states at every sub-batch (to consider TBPTT)
        self.hidden = None
        self.cell = None
        self.att_weights = None
        self.att_hidden = None

        self.prev_kappa = 0
        self.prev_ctx = None

        self.null_state = True

    def reset_bias(self):
        # need to reset bias at every iteration to avoid unnecessary computation
        self.att_bias_enc = None
        self.att_bias_spkr = None

        self.null_bias = True

        self.att_bias_style = None

    def mask_states(self, len_mask, debug=0):
        if not self.null_state:
            if len_mask is None:
                self.hidden = self.hidden.data
                self.cell = self.cell.data
                self.att_weights = self.att_weights.data
                self.att_hidden = self.att_hidden.data
            else:
                self.hidden = torch.index_select(self.hidden.data, 1, len_mask).data
                self.cell = torch.index_select(self.cell.data, 1, len_mask).data
                self.att_weights = torch.index_select(self.att_weights.data, 0, len_mask).data
                self.att_hidden = torch.index_select(self.att_hidden.data, 0, len_mask).data

    def get_velocity_loss(self):
        loss = self.velocity_loss
        self.velocity_loss = 0
        return loss