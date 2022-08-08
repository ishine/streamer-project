import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AttnDecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, r_factor=2, dropout_p=0.5, debug=0, **kwargs):
        super(AttnDecoderRNN, self).__init__()
        self.r_factor = r_factor
        self.H_dec = dec_hidden
        self.O_dec = output_size
        self.num_lstm_layers = 2
        self.exp_no = kwargs.get('exp_no', 0)
        self.tside_prosody_size = kwargs.get('tside_prosody_size', 0)
        self.sside_prosody_size = kwargs.get('sside_prosody_size', 0)
        
        self.speed_proj = nn.Sequential(
            nn.Linear(1, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dec_hidden, enc_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_p)
        )

        prenet_out_dim = dec_hidden
        self.prenet = nn.Sequential(
            nn.Linear(output_size, 2 * dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * dec_hidden, prenet_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )   
        in_lstm_size = enc_hidden - self.tside_prosody_size + self.sside_prosody_size + prenet_out_dim + 1

        self.spkr_proj = nn.Sequential(
            nn.Linear(spkr_embed_size, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dec_hidden, in_lstm_size),
            nn.Tanh(),
            nn.Dropout(dropout_p)
        )
        self.LSTM = nn.LSTM(in_lstm_size, 4 * dec_hidden, num_layers=self.num_lstm_layers, batch_first=True)

        # self.prosody_prior = nn.GRU(enc_hidden + dec_hidden + self.sside_prosody_size, dec_hidden, batch_first=True)
        # self.prosody_proj = nn.Linear(dec_hidden, self.sside_prosody_size)

        self.out_linear = nn.Linear(enc_hidden - self.tside_prosody_size + self.sside_prosody_size + 4 * dec_hidden, output_size * r_factor)

        self.set_attention_range(att_range)
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, lengths_enc, speed, debug=0, **kwargs):
        N, T_enc = spkr_vec.size(0), lengths_enc.max().item()
        curr_sside_prosody_ref = kwargs.get('curr_sside_prosody_ref')
        self.att_weights = kwargs.get('attention_ref')
        att_position = kwargs.get('att_position')
        is_aug = kwargs.get('is_aug', False)

        if is_aug:
            hidden, cell = self.hidden2, self.cell2
        else:
            hidden, cell = self.hidden, self.cell

        if self.null_state:
            self.null_state = False

        out_prenet = self.prenet(input_dec)  # N x O_dec -> N x 1 x H
        
        att_transpose = self.att_weights.transpose(1, 2)
        context = torch.bmm(att_transpose, input_enc)                    # N x 1 x 2H - P

        ####### tside prediction ######
        # prev_sside_prosody_ref = kwargs.get('prev_sside_prosody_ref')
        # tside_prosody_pred = kwargs.get('tside_prosody_pred')
        # tside_prosody_context = torch.bmm(att_transpose, tside_prosody_pred)     # N x 1 x P
        #
        # if speed is not None:
        #     sside_pred_input = sside_pred_input + self.speed_proj(speed.unsqueeze(-1)).unsqueeze(1)
        #
        # # set sside prosody
        # if prev_sside_prosody_ref is None:
        #     prev_sside_prosody_ref = sside_pred_input.data.new().resize_(N, 1, self.sside_prosody_size).zero_()
        #
        # prev_sside_prosody_ref = prev_sside_prosody_ref.detach()
        # pronunciation_context = context.detach()
        # additional_ar_info = out_prenet.detach()
        # in_prosody_pred = torch.cat([pronunciation_context, additional_ar_info, tside_prosody_context, prev_sside_prosody_ref, spkr_vec], dim=2)
        # curr_sside_prosody_pred, self.prosody_hidden = self.prosody_prior(in_prosody_pred, self.prosody_hidden)
        # curr_sside_prosody_pred = self.prosody_proj(curr_sside_prosody_pred)
        ####### tside prediction ######

        curr_sside_prosody_pred = None
        if curr_sside_prosody_ref is None:
            sside_prosody = curr_sside_prosody_pred                          # N x 1 x P
        else:
            sside_prosody = curr_sside_prosody_ref                           # N x 1 x P
            
        # concat sequential prosody
        in_lstm = torch.cat((out_prenet, context, att_position, sside_prosody), 2)         # N x 1 x 4H
        in_lstm += self.spkr_proj(spkr_vec)
        if hidden is None:
            out_lstm, (hidden, cell) = self.LSTM(in_lstm, None)           # N x T_dec x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (hidden, cell) = self.LSTM(in_lstm, (hidden, cell))  # N x T_dec x 4H, L x N x 4H, L x N x 4H
        if is_aug:
            self.hidden2, self.cell2 = hidden, cell
        else:
            self.hidden, self.cell = hidden, cell

        dec_output = torch.cat((out_lstm, context, sside_prosody), 2)                              # N x 1 x 6H

        output = self.out_linear(dec_output).view(N, -1, self.O_dec)                 # N x r x O_dec
        return {
            "output_dec": output,
            "context": context,
            "output_lstm": dec_output,
            "sside_prosody_pred": curr_sside_prosody_pred,
            "output_prenet": out_prenet,
        }

    def set_attention_range(self, r):
        self.att_range = r

    def reset_states(self, debug=0):
        # need to reset states at every sub-batch (to consider TBPTT)
        self.hidden = None
        self.cell = None
        self.hidden2 = None
        self.cell2 = None
        self.att_weights = None
        # self.prosody_hidden = None

        self.null_state = True

    def mask_states(self, len_mask, debug=0):
        if not self.null_state:
            if len_mask is None:
                self.hidden = self.hidden.data
                self.cell = self.cell.data
                if self.hidden2 is not None:
                    self.hidden2 = self.hidden2.data
                    self.cell2 = self.cell2.data
                # self.prosody_hidden = self.prosody_hidden.data
            else:
                self.hidden  = torch.index_select(self.hidden.data, 1, len_mask).data
                self.cell  = torch.index_select(self.cell.data, 1, len_mask).data
                if self.hidden2 is not None:
                    self.hidden2 = torch.index_select(self.hidden2.data, 1, len_mask).data
                    self.cell2 = torch.index_select(self.cell2.data, 1, len_mask).data
                # self.prosody_hidden  = torch.index_select(self.prosody_hidden.data, 1, len_mask).data

def PositionalEncoding(d_model, lengths, w_s=None):
    L = int(lengths.max().item())
    if w_s is None:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0).to(lengths.device)
    else:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0).to(lengths.device) * w_s.unsqueeze(-1)
    div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model).to(lengths.device)
    pe = torch.zeros(len(lengths), L, d_model).to(lengths.device)
    
    pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) / div_term.unsqueeze(0))
    pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) / div_term.unsqueeze(0))
    return pe


##################### FAST SPEECH #########################
from tacotron.module.fastspeech.Conformer import ConformerBlock
from tacotron.module.fastspeech.Layers import FFTBlock
from tacotron.module.fastspeech.SubLayers import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask

class FastSpeechDecoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 enc_hidden,
                 dec_hidden,
                 dec_out_size,
                 spkr_embed_size,
                 r_factor,
                 sside_prosody_size,
                 n_layers=4,
                 n_head=2,
                 len_max_seq=3000,
                 d_model=256,
                 d_inner=1024,
                 dropout=0.1,
                 debug=0,
                 ):

        super(FastSpeechDecoder, self).__init__()

        n_position = len_max_seq + 1
        self.n_layers = n_layers
        self.O_dec = dec_out_size

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.enc_proj = nn.Linear(enc_hidden, d_model - sside_prosody_size)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, dec_hidden, dec_hidden, dropout=dropout) for _ in range(n_layers)])

        # speaker biases will be sliced.
        self.speaker_proj = nn.Linear(spkr_embed_size, d_model * n_layers)

        self.out_proj = nn.Linear(d_model, dec_out_size * r_factor)

    def forward(self, enc_seq, enc_pos, spkr_vec, attention=None, att_position=None, return_attns=False, **kwargs):
        N = enc_seq.size(0)
        dec_slf_attn_list = []
        curr_sside_prosody_ref = kwargs.get('curr_sside_prosody_ref')

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Speaker bias
        spkr_biases = self.speaker_proj(spkr_vec)    # N x 1 x nH
        spkr_biases_list = torch.chunk(spkr_biases, self.n_layers, dim=2)

        # -- Sequential prosody
        curr_sside_prosody_pred = None
        if curr_sside_prosody_ref is None:
            sside_prosody = curr_sside_prosody_pred                          # N x 1 x P
        else:
            sside_prosody = curr_sside_prosody_ref                           # N x 1 x P

        # -- Forward
        if attention is not None:
            context = torch.bmm(attention.transpose(1,2), self.enc_proj(enc_seq))
        else:
            context = self.enc_proj(enc_seq)

        context = torch.cat([context, sside_prosody], dim=2)
        if att_position is not None:
            dec_output = context + self.position_enc(enc_pos) + self.att_position_proj(att_position)
        else:
            dec_output = context + self.position_enc(enc_pos)

        for i, dec_layer in enumerate(self.layer_stack):
            dec_output += spkr_biases_list[i]
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                dec_output,
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        
        output = self.out_proj(dec_output).view(N, -1, self.O_dec)
        return {
            "output_dec": output,
            "context": context,
            "output_lstm": None,
        }

    def reset_states(self, debug=0):
        pass

    def mask_states(self, len_mask, debug=0):
        pass
##################### FAST SPEECH #########################

