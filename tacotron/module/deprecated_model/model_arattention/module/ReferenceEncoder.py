import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import numpy as np


class ReferenceEncoder(nn.Module):
    def __init__(self, input_hidden_dim, style_dim, att_dim, n_token, debug=0, n_head=-1, spkr_embed_size=-1):
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

        gru_out_dim = att_dim
        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 6)))
        self.GRU = nn.GRU(input_size=gru_in_dim, hidden_size=gru_out_dim, num_layers=2, batch_first=True)

        self.n_head = n_head
        if n_head > 0:
            total_dim = style_dim
            assert total_dim % n_head == 0

            segment_dim = total_dim // n_head

            self.n_token = n_token
            self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, total_dim)))
            self.register_parameter('token_keys', nn.Parameter(torch.randn(1, n_token, total_dim)))
            self.ref_proj = nn.Linear(gru_out_dim, total_dim, bias=True)
            self.key_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.query_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.token_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.style_proj = nn.Linear(total_dim, total_dim, bias=False)
        else:
            self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, style_dim)))
            self.ref_proj = nn.Linear(gru_out_dim, att_dim, bias=True)
            self.token_proj = nn.Linear(style_dim, att_dim, bias=False)
            self.att_proj = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x, whole_spec_len=None, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T

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
            output = rnn.pack_padded_sequence(output, adjusted_spec_len, True, enforce_sorted=False)
            output, _ = self.GRU(output)                                    # N x T x H_ref
            output, _ = rnn.pad_packed_sequence(output, True)               # NxTx2H
            H_out = output.size(-1)
            ref_encoding = torch.gather(output, 1, (adjusted_spec_len-1).view(N, 1, 1).expand(-1, -1, H_out))

        if self.n_head > 0:
            K, h = self.n_token, self.n_head
            # attention is all you need
            token_bank = self.token_bank.view(1, K, h, -1)                  # 1 x T_tok x h x H_seg
            token_keys = self.token_keys.view(1, K, h, -1)                  # 1 x T_tok x h x H_seg
            query = self.ref_proj(ref_encoding).view(N, 1, h, -1)           # N x 1 x h x H_seg

            projed_value = self.token_proj(token_bank).transpose(1,2)               # 1 x h x T_tok x H_seg
            projed_key = self.key_proj(token_keys).transpose(1,2).transpose(2,3)    # 1 x h x H_seg x T_tok
            projed_query = self.query_proj(query).transpose(1,2)                    # N x h x 1 x H_seg

            # stable softmax
            logit = torch.matmul(projed_query, projed_key) / np.sqrt(projed_key.size(-1))  # N x h x 1 x T_tok
            logit_max, _ = torch.max(logit, dim=3, keepdim=True)
            att_weights = torch.exp(logit - logit_max)
            att_weights = F.normalize(att_weights, 1, 3)                    # N x h x 1 x T_tok

            style_vec = torch.matmul(att_weights, projed_value)             # N x h x 1 x H_seg
            style_vec = style_vec.transpose(1,2).view(N, 1, -1)             # N x 1 x H_sty
            style_vec = torch.tanh(self.style_proj(style_vec))              # N x 1 x H_sty
        else:
            token_bank = self.token_bank.expand(N, -1, -1)  # N x T_tok x H_sty

            # attention -- https://arxiv.org/pdf/1506.07503.pdf
            e = self.token_proj(token_bank) + self.ref_proj(ref_encoding)       # N x T_tok x H_att

            # stable softmax
            logit = self.att_proj(torch.tanh(e))
            logit_max, _ = torch.max(logit, dim=1, keepdim=True)
            att_weights = torch.exp(logit - logit_max)
            att_weights = F.normalize(att_weights, 1, 1)              # N x T_tok x 1
            style_vec = torch.bmm(att_weights.transpose(1, 2), torch.tanh(token_bank))   # N x 1 x H_sty

        return style_vec, att_weights

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


class ReferenceEncoderTside(nn.Module):
    def __init__(self, input_hidden_dim, style_dim, key_dim, debug=0, spkr_embed_size=-1, **kwargs):
        super(ReferenceEncoderTside, self).__init__()
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
        self.stride_size_ref = [(2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1)]

        self.conv_ref_enc = nn.ModuleList()
        for c, f, s in zip(self.out_channel_ref, self.filter_size_ref, self.stride_size_ref):
            self.conv_ref_enc.append(get_conv_bn(in_channel, c, f, stride=s))
            in_channel = c

        self.spkr_biases = nn.ModuleList()
        for c in self.out_channel_ref:
            self.spkr_biases.append(nn.Linear(spkr_embed_size, c, bias=False))

        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 2)))
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

        ref_out_dict.update({
            "tside_prosody_vec": output[:, :, :self.style_dim],
            "tside_att_key_vec": output[:, :, self.style_dim:]
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


class ReferenceEncoderOld(nn.Module):
    # Deprecated.
    def __init__(self, input_hidden_dim, style_dim, att_dim, n_token, debug=0, n_head=-1, spkr_embed_size=-1):
        super(ReferenceEncoderOld, self).__init__()
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

        gru_out_dim = att_dim
        gru_in_dim = self.out_channel_ref[-1] * int(np.ceil(input_hidden_dim / (2 ** 6)))
        self.GRU = nn.GRU(input_size=gru_in_dim, hidden_size=gru_out_dim, num_layers=2, batch_first=True)

        self.n_head = n_head
        if n_head > 0:
            total_dim = style_dim
            assert total_dim % n_head == 0

            segment_dim = total_dim // n_head

            self.n_token = n_token
            self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, total_dim)))
            self.register_parameter('token_keys', nn.Parameter(torch.randn(1, n_token, total_dim)))
            self.ref_proj = nn.Linear(gru_out_dim, total_dim, bias=True)
            self.key_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.query_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.token_proj = nn.Linear(segment_dim, segment_dim, bias=False)
            self.style_proj = nn.Linear(total_dim, total_dim, bias=False)
        else:
            self.register_parameter('token_bank', nn.Parameter(torch.randn(1, n_token, style_dim)))
            self.ref_proj = nn.Linear(gru_out_dim, att_dim, bias=True)
            self.token_proj = nn.Linear(style_dim, att_dim, bias=False)
            self.att_proj = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T

        T_out = output_ref_enc.size(-1)
        output = output_ref_enc.view(N, -1, T_out).transpose(1, 2)
        output, _ = self.GRU(output.contiguous())                           # N x T x H_ref
        ref_encoding = output[:, -1:]                                       # N x 1 x H_ref

        if self.n_head > 0:
            K, h = self.n_token, self.n_head
            # attention is all you need
            token_bank = self.token_bank.view(1, K, h, -1)                  # 1 x T_tok x h x H_seg
            token_keys = self.token_keys.view(1, K, h, -1)                  # 1 x T_tok x h x H_seg
            query = self.ref_proj(ref_encoding).view(N, 1, h, -1)           # N x 1 x h x H_seg

            projed_value = self.token_proj(token_bank).transpose(1,2)               # 1 x h x T_tok x H_seg
            projed_key = self.key_proj(token_keys).transpose(1,2).transpose(2,3)    # 1 x h x H_seg x T_tok
            projed_query = self.query_proj(query).transpose(1,2)                    # N x h x 1 x H_seg

            # stable softmax
            logit = torch.matmul(projed_query, projed_key) / np.sqrt(projed_key.size(-1))  # N x h x 1 x T_tok
            logit_max, _ = torch.max(logit, dim=3, keepdim=True)
            att_weights = torch.exp(logit - logit_max)
            att_weights = F.normalize(att_weights, 1, 3)                    # N x h x 1 x T_tok

            # modification
            # selected_idx = 1
            # att_weights = att_weights[selected_idx:selected_idx+1].expand(N,-1,-1,-1)

            style_vec = torch.matmul(att_weights, projed_value)             # N x h x 1 x H_seg
            style_vec = style_vec.transpose(1,2).view(N, 1, -1)             # N x 1 x H_sty
            style_vec = torch.tanh(self.style_proj(style_vec))              # N x 1 x H_sty
        else:
            token_bank = self.token_bank.expand(N, -1, -1)  # N x T_tok x H_sty

            # attention -- https://arxiv.org/pdf/1506.07503.pdf
            e = self.token_proj(token_bank) + self.ref_proj(ref_encoding)       # N x T_tok x H_att

            # stable softmax
            logit = self.att_proj(torch.tanh(e))
            logit_max, _ = torch.max(logit, dim=1, keepdim=True)
            att_weights = torch.exp(logit - logit_max)
            att_weights = F.normalize(att_weights, 1, 1)              # N x T_tok x 1

            # modification
            # selected_idx = 5
            # att_weights = att_weights[selected_idx:selected_idx+1].expand(N,-1,-1)

            style_vec = torch.bmm(att_weights.transpose(1, 2), torch.tanh(token_bank))   # N x 1 x H_sty

        # if not self.training:
        #     print(att_weights.view(N, -1).numpy())

        return style_vec, att_weights

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
