import torch
import torch.nn as nn
from tacotron.module.fastspeech.Layers import FFTBlock
from tacotron.module.fastspeech.Conformer import ConformerBlock
from tacotron.module.fastspeech.SubLayers import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask

class FastSpeechDecoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 enc_hidden,
                 dec_hidden,
                 dec_out_size,
                 spkr_embed_size,
                 r_factor,
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

        self.enc_proj = nn.Linear(enc_hidden, d_model)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, dec_hidden, dec_hidden, dropout=dropout) for _ in range(n_layers)])

        # speaker biases will be sliced.
        self.speaker_proj = nn.Linear(spkr_embed_size, d_model * n_layers)

        self.out_proj = nn.Linear(d_model, dec_out_size * r_factor)

    def forward(self, enc_seq, enc_pos, spkr_vec=None, attention=None, att_position=None, return_attns=False):
        N = enc_seq.size(0)
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Speaker bias
        if spkr_vec is None:
            spkr_biases_list = [0 for _ in range(len(self.layer_stack))]
        else:
            spkr_biases = self.speaker_proj(spkr_vec)    # N x 1 x nH
            spkr_biases_list = torch.chunk(spkr_biases, self.n_layers, dim=2)

        # -- Forward
        if attention is not None:
            context = torch.bmm(attention.transpose(1,2), self.enc_proj(enc_seq))
        else:
            context = self.enc_proj(enc_seq)

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

    def reset_bias(self):
        pass

    def mask_states(self, len_mask, debug=0):
        pass
