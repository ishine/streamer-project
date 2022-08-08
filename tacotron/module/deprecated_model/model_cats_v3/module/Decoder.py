import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, dec_hidden, output_size, spkr_embed_size, pitch_size=1, pitch_embed_size=0, dropout_p=0.5, debug=0):
        super(DecoderRNN, self).__init__()
        self.H_dec = dec_hidden
        self.O_dec = output_size
        self.num_lstm_layers = 2

        self.prenet = nn.Sequential(
            nn.Linear(output_size, 2 * dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        if pitch_embed_size > 0:
            self.pitch_emb = nn.Linear(pitch_size, pitch_embed_size)

        in_lstm_size = enc_hidden + dec_hidden + spkr_embed_size + pitch_embed_size + 1
        self.LSTM = nn.LSTM(
            in_lstm_size,
            3 * dec_hidden,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )

        self.out_linear = nn.Linear(
            enc_hidden + 3 * dec_hidden,
            output_size
        )
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, debug=0, **kwargs):
        N = input_enc.size(0)
        attention_ref = kwargs.get('attention_ref')
        att_position = kwargs.get('att_position')
        pitch = kwargs.get('pitch')
        sside_prosody = kwargs.get('sside_prosody')
        is_aug = kwargs.get('is_aug', False)

        if is_aug:
            hidden, cell = self.hidden2, self.cell2
        else:
            hidden, cell = self.hidden, self.cell

        if self.null_state:
            self.null_state = False

        if input_dec is None:
            input_dec = torch.zeros(N, 1, self.O_dec, device=input_enc.device, dtype=torch.float)
        out_prenet = self.prenet(input_dec)  # N x T_dec x H        

        context = torch.bmm(attention_ref.transpose(1, 2), input_enc)   # N x T_dec x 2H
        spkr_vec_expanded = spkr_vec.expand(-1, out_prenet.size(1), -1)
        
        if pitch is not None:
            pitch_flag = torch.ne(pitch, 0).float()
            expanded_pitch = self.pitch_emb(pitch) * pitch_flag     # N x T_enc x 1
            in_lstm = torch.cat((out_prenet, context, spkr_vec_expanded, att_position, expanded_pitch), 2)        # N x T_dec x 4H + 1
        elif sside_prosody is not None:
            # no flag is needed for sside_prosody
            sside_prosody = self.pitch_emb(sside_prosody)                   # N x T_enc x 1
            in_lstm = torch.cat((out_prenet, context, spkr_vec_expanded, att_position, sside_prosody), 2)        # N x T_dec x 4H + 1
        else:
            in_lstm = torch.cat((out_prenet, context, spkr_vec_expanded, att_position), 2)        # N x T_dec x 4H + 1
        if hidden is None:
            out_lstm, (hidden, cell) = self.LSTM(in_lstm, None)           # N x T_dec x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (hidden, cell) = self.LSTM(in_lstm, (hidden, cell))  # N x T_dec x 4H, L x N x 4H, L x N x 4H
        if is_aug:
            self.hidden2, self.cell2 = hidden, cell
        else:
            self.hidden, self.cell = hidden, cell

        dec_output = torch.cat((out_lstm, context), 2)                              # N x T_dec x 6H

        output = self.out_linear(dec_output).view(N, -1, self.O_dec)                 # N x r x O_dec
        return {
            "output_dec": output,
            "context": context,
            "output_lstm": dec_output,
            "output_prenet": out_prenet,
        }

    def reset_states(self, debug=0):
        # need to reset states at every sub-batch (to consider TBPTT)
        self.hidden = None
        self.cell = None
        self.hidden2 = None
        self.cell2 = None
        self.null_state = True

    def mask_states(self, len_mask, debug=0):
        if not self.null_state:
            if len_mask is None:
                self.hidden = self.hidden.data
                self.cell = self.cell.data
                if self.hidden2 is not None:
                    self.hidden2 = self.hidden2.data
                    self.cell2 = self.cell2.data
            else:
                self.hidden = torch.index_select(self.hidden.data, 1, len_mask).data
                self.cell = torch.index_select(self.cell.data, 1, len_mask).data
                if self.hidden2 is not None:
                    self.hidden2 = torch.index_select(self.hidden2.data, 1, len_mask).data
                    self.cell2 = torch.index_select(self.cell2.data, 1, len_mask).data
