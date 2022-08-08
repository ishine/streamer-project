import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import numpy as np


class ReferenceEncoder(nn.Module):
    def __init__(self, input_hidden_dim, style_dim, att_dim, n_token, debug=0, spkr_embed_size=-1):
        super(ReferenceEncoder, self).__init__()
        self.style_dim = style_dim

        def get_conv_bn(in_dim, out_dim, kernel_size, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        in_channel = 1
        self.out_channel_ref = [32, 32, 64, 64, 128, 128]
        self.filter_size_ref = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        self.stride_size_ref = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        self.conv_ref_enc = nn.ModuleList()
        for c, f, s in zip(self.out_channel_ref, self.filter_size_ref, self.stride_size_ref):
            self.conv_ref_enc.append(get_conv_bn(in_channel, c, f, stride=s))
            in_channel = c

        self.spkr_biases = nn.ModuleList()
        for c in self.out_channel_ref:
            self.spkr_biases.append(nn.Linear(spkr_embed_size, c, bias=False))

        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 6)))
        gru_out_dim = att_dim
        self.GRU = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_out_dim,
            num_layers=2,
            batch_first=True
        )

        self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, style_dim)))
        self.ref_proj = nn.Linear(gru_out_dim, att_dim, bias=True)
        self.token_proj = nn.Linear(style_dim, att_dim, bias=False)
        self.att_proj = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x, whole_spec_len=None, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size()
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T
        ref_out_dict = {}

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)           # N x H2 x C x T
            output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

        T_out = output_ref_enc.size(-1)
        output = output_ref_enc.view(N, -1, T_out).transpose(1, 2).contiguous()
        if whole_spec_len is None:
            output, _ = self.GRU(output)                                    # N x T x H_ref
            ref_encoding = output[:, -1:]                                   # N x 1 x H_ref
        else:
            downsample_factor = 2 ** 6
            adjusted_spec_len = torch.ceil(whole_spec_len.float() / downsample_factor).long()
            output = rnn.pack_padded_sequence(output, adjusted_spec_len.cpu(), True, enforce_sorted=False)
            output, _ = self.GRU(output)                                    # N x T x H_ref
            output, _ = rnn.pad_packed_sequence(output, True)               # NxTx2H
            H_out = output.size(-1)
            ref_encoding = torch.gather(output, 1, (adjusted_spec_len-1).view(N, 1, 1).expand(-1, -1, H_out))

        token_bank = self.token_bank.expand(N, -1, -1)  # N x T_tok x H_sty

        # attention -- https://arxiv.org/pdf/1506.07503.pdf
        e = self.token_proj(token_bank) + self.ref_proj(ref_encoding)       # N x T_tok x H_att
        logit = self.att_proj(torch.tanh(e))
        att_weights = F.softmax(logit, dim=1)
        style_vec = torch.bmm(att_weights.transpose(1, 2), torch.tanh(token_bank))   # N x 1 x H_sty

        ref_out_dict.update({
            'gst': style_vec,
            'att_weights': att_weights,
        })
        return ref_out_dict

    def get_style_token(self, style_idx):
        token = self.token_bank[:, style_idx:style_idx+1]                   # 1 x T_tok x H_sty
        return torch.tanh(token)

    def pad_SAME(self, x, filter_size, stride):
        in_height, in_width = x.size(-2), x.size(-1)
        if (in_height % stride[0] == 0):
            pad_along_height = max(filter_size[0] - stride[0], 0)
        else:
            pad_along_height = max(filter_size[0] - (in_height % stride[0]), 0)
        if (in_width % stride[1] == 0):
            pad_along_width = max(filter_size[1] - stride[1], 0)
        else:
            pad_along_width = max(filter_size[1] - (in_width % stride[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class ReferenceEncoderSside(nn.Module):
    def __init__(self, input_hidden_dim, style_dim, key_dim, r_factor, debug=0, spkr_embed_size=-1, **kwargs):
        super(ReferenceEncoderSside, self).__init__()
        self.style_dim = style_dim
        self.exp_no = kwargs.get('exp_no')
        self.debug = kwargs.get('debug')

        def get_conv_bn(in_dim, out_dim, kernel_size, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        in_channel = 1
        self.out_channel_ref = [32, 32, 64, 64, 128, 128]
        self.filter_size_ref = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        if r_factor == 1:
            self.stride_size_ref = [(2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1)]
        elif r_factor == 2:
            self.stride_size_ref = [(2, 2), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1)]
        elif r_factor == 4:
            self.stride_size_ref = [(2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1)]

        self.conv_ref_enc = nn.ModuleList()
        for c, f, s in zip(self.out_channel_ref, self.filter_size_ref, self.stride_size_ref):
            self.conv_ref_enc.append(get_conv_bn(in_channel, c, f, stride=s))
            in_channel = c

        self.spkr_biases = nn.ModuleList()
        for c in self.out_channel_ref:
            self.spkr_biases.append(nn.Linear(spkr_embed_size, c, bias=False))

        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 2)))
        if self.debug == 30 or self.debug == 31 or self.debug == 32 or self.debug == 33 or self.debug == 36 \
             or self.debug == 37 or self.debug == 38 or self.debug == 39 or self.debug == 40 or self.debug == 41 \
             or self.debug == 42 or self.debug == 43 or self.debug == 44 or self.debug == 45 or self.debug == 46 or self.debug == 47:
            gru_out_dim = input_hidden_dim
            self.GRU = nn.GRU(input_size=gru_in_dim, hidden_size=gru_out_dim, num_layers=2, batch_first=True)
            self.out_proj = nn.Linear(gru_out_dim, style_dim)
        else:
            gru_out_dim = style_dim + key_dim
            self.GRU = nn.GRU(input_size=gru_in_dim, hidden_size=gru_out_dim, num_layers=2, batch_first=True)

    def forward(self, x, spkr_vec, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            seq_style_vec: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T
        ref_out_dict = {}

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

        T_out = output_ref_enc.size(-1)
        output = output_ref_enc.view(N, -1, T_out).transpose(1, 2)
        output, _ = self.GRU(output.contiguous())                           # N x T x H_ref

        if self.debug == 30 or self.debug == 31 or self.debug == 32 or self.debug == 33 or self.debug == 36 \
             or self.debug == 37 or self.debug == 38 or self.debug == 39 or self.debug == 40 or self.debug == 41 \
             or self.debug == 42 or self.debug == 43 or self.debug == 44 or self.debug == 45 or self.debug == 46 or self.debug == 47:
            output = self.out_proj(output)
            if self.debug == 33 or self.debug == 37:
                output = torch.tanh(output)

        ref_out_dict.update({
            "sside_prosody_vec": output[:, :, :self.style_dim],
            "sside_att_key_vec": output[:, :, self.style_dim:]
        })
        return ref_out_dict

    def pad_SAME(self, x, filter_size, stride):
        in_height, in_width = x.size(-2), x.size(-1)
        if (in_height % stride[0] == 0):
            pad_along_height = max(filter_size[0] - stride[0], 0)
        else:
            pad_along_height = max(filter_size[0] - (in_height % stride[0]), 0)
        if (in_width % stride[1] == 0):
            pad_along_width = max(filter_size[1] - stride[1], 0)
        else:
            pad_along_width = max(filter_size[1] - (in_width % stride[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class ReferenceEncoderIntegrated(nn.Module):
    def __init__(self, input_hidden_dim, style_dim, gst_dim, att_hidden, n_token, debug=0, spkr_embed_size=-1, **kwargs):
        super(ReferenceEncoderIntegrated, self).__init__()
        self.style_dim = style_dim
        self.exp_no = kwargs.get('exp_no')

        def get_conv_bn(in_dim, out_dim, kernel_size, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        # for prosody extraction
        in_channel = 1
        self.out_channel_ref = [32, 32, 64, 64, 128, 128]
        self.filter_size_ref = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        self.stride_size_ref = [(2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1)]

        self.conv_ref_enc = nn.ModuleList()
        for c, f, s in zip(self.out_channel_ref, self.filter_size_ref, self.stride_size_ref):
            self.conv_ref_enc.append(get_conv_bn(in_channel, c, f, stride=s))
            in_channel = c

        self.spkr_biases = nn.ModuleList()
        for c in self.out_channel_ref:
            self.spkr_biases.append(nn.Linear(spkr_embed_size, c, bias=False))

        # speech-side prosody
        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 2)))
        gru_out_dim = att_hidden
        self.GRU = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_out_dim,
            num_layers=2,
            batch_first=True
        )
        self.out_proj = nn.Linear(att_hidden, style_dim)

        # for GST
        self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, gst_dim)))
        self.ref_proj = nn.Linear(gru_out_dim, att_hidden, bias=True)
        self.token_proj = nn.Linear(gst_dim, att_hidden, bias=False)
        self.att_proj = nn.Linear(att_hidden, 1, bias=False)

    def forward(self, x, whole_spec_len, spkr_vec, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            seq_style_vec: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size()
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T
        ref_out_dict = {}

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

        T_out = output_ref_enc.size(-1)
        conv_output = output_ref_enc.view(N, -1, T_out).transpose(1, 2)
        output = rnn.pack_padded_sequence(conv_output, whole_spec_len, True, enforce_sorted=False)
        output, _ = self.GRU(output)                           # N x T x H_ref
        gru_output, _ = rnn.pad_packed_sequence(output, True)               # NxTx2H

        # GST
        H_out = gru_output.size(-1)
        ref_encoding = torch.gather(gru_output, 1, (whole_spec_len-1).view(N, 1, 1).expand(-1, -1, H_out).long())
        token_bank = self.token_bank.expand(N, -1, -1)  # N x T_tok x H_sty
        e = self.token_proj(token_bank) + self.ref_proj(ref_encoding)       # N x T_tok x H_att
        logit = self.att_proj(torch.tanh(e))
        att_weights = F.softmax(logit, dim=1)
        style_vec = torch.bmm(att_weights.transpose(1, 2), torch.tanh(token_bank))   # N x 1 x H_sty

        # speech-side prosody
        sside_prosody = self.out_proj(gru_output)       # N x T_dec x H

        ref_out_dict.update({
            'gst': style_vec,
            'sside_prosody_vec': sside_prosody[:, :, :self.style_dim],
            'sside_att_key_vec': sside_prosody[:, :, self.style_dim:],
            'seq_embedding': style_vec,
        })
        return ref_out_dict

    def pad_SAME(self, x, filter_size, stride):
        in_height, in_width = x.size(-2), x.size(-1)
        if (in_height % stride[0] == 0):
            pad_along_height = max(filter_size[0] - stride[0], 0)
        else:
            pad_along_height = max(filter_size[0] - (in_height % stride[0]), 0)
        if (in_width % stride[1] == 0):
            pad_along_width = max(filter_size[1] - stride[1], 0)
        else:
            pad_along_width = max(filter_size[1] - (in_width % stride[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class SpeechEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
