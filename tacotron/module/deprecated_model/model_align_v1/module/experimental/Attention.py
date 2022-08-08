import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import numpy as np

from tacotron.module.deprecated_model.model_align_v1.module.experimental.commons import maximum_path

class Attention(nn.Module):
    def __init__(self, enc_hidden, mel_dim, att_hidden, r_factor, debug=0, dropout_p=0.5, **kwargs):
        super(Attention, self).__init__()
        def get_conv_bn(in_dim, out_dim, kernel_size, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.r_factor = r_factor
        self.att_upsample = True
        self.enc_proj = nn.Linear(enc_hidden, att_hidden, bias=True)

        in_channel = 1
        self.out_channel_ref = [32, 64, 128]
        self.filter_size_ref = [(3, 3), (3, 3), (3, 3)]
        if r_factor == 1:
            self.stride_size_ref = [(2, 1), (2, 1), (1, 1)]
        elif r_factor == 2:
            self.stride_size_ref = [(2, 2), (2, 1), (1, 1)]
        elif r_factor == 4:
            self.stride_size_ref = [(2, 2), (2, 2), (1, 1)]

        self.conv_ref_enc = nn.ModuleList()
        for c, f, s in zip(self.out_channel_ref, self.filter_size_ref, self.stride_size_ref):
            self.conv_ref_enc.append(get_conv_bn(in_channel, c, f, stride=s))
            in_channel = c

        spec_proj_in_dim = self.out_channel_ref[-1] * int(np.ceil(mel_dim / (2 ** 2)))
        self.query_proj = nn.Linear(spec_proj_in_dim, att_hidden)

        self.sim_proj = nn.Linear(1, 1)

    def forward(self, text, spec, text_lengths, spec_lengths, debug=0):
        """ spec: N x T x O_dec sized Tensor (Spectrogram)
            seq_style_vec: N x (T/r_factor) x H sized Tensor
        """
        N, T_dec, _ = spec.size()

        # from text
        key = self.enc_proj(text)                               # N x T_enc x H_att

        # from spectrogram
        enc_spec_mask = torch.arange(0, T_dec, device=key.device).view(1, -1).expand(N, -1)
        enc_spec_mask = torch.lt(enc_spec_mask, spec_lengths.view(-1, 1).expand(-1, T_dec))                 # N x T_dec
        enc_spec_mask = enc_spec_mask.view(N, 1, -1, T_dec)
        output_ref_enc = spec.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T
        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = output_ref_enc * enc_spec_mask
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            enc_spec_mask = enc_spec_mask[:, :, :, ::self.stride_size_ref[i][1]]

        T_out = output_ref_enc.size(-1)
        output = output_ref_enc.view(N, -1, T_out).transpose(1, 2)
        query = self.query_proj(output.contiguous())                           # N x T_dec x H_att

        if self.att_upsample and self.r_factor != 4:
            if query.size(1) % 4 != 0:
                query = torch.cat([query[:, np.random.randint(0,4)::4], query[:, -1:]], dim=1)
            else:
                query = query[:, np.random.randint(0,4)::4]
            spec_length_factor = 4
        else:
            spec_length_factor = 1

        # masking
        with torch.no_grad():
            text_mask = torch.arange(0, key.size(1), device=key.device).view(1, -1).expand(N, -1)
            text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, key.size(1)))                     # N x T_enc
            spec_lengths = spec_lengths // (self.r_factor * spec_length_factor)
            spec_mask = torch.arange(0, query.size(1), device=key.device).view(1, -1).expand(N, -1)
            spec_mask = torch.lt(spec_mask, spec_lengths.view(-1, 1).expand(-1, query.size(1)))                 # N x T_dec
            att_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(1)                                      # N x T_enc x T_dec    

        key = key / torch.norm(key, p=2, dim=2, keepdim=True)
        query = query / torch.norm(query, p=2, dim=2, keepdim=True)
        logit = torch.bmm(key, query.transpose(1, 2))              # N x T_enc x T_dec
        logit_proj = self.sim_proj(logit.unsqueeze(-1)).squeeze(-1)

        # hard att
        with torch.no_grad():
            attention = maximum_path(torch.exp(logit_proj), att_mask).detach()

        # attention penalty (for stability, check: https://stackoverflow.com/questions/44081007/logsoftmax-stability)
        att_loss = 0
        match_mask = attention

        inter_class_similarity_max, _ = torch.max(logit_proj, dim=1, keepdim=True)
        inter_class_similarity_net = logit_proj - inter_class_similarity_max
        inter_class_similarity_net_exp = torch.exp(inter_class_similarity_net)
        inter_class_loss1 = spec_mask * (
            - torch.sum(inter_class_similarity_net * match_mask, dim=1) \
            + torch.log(torch.clamp(torch.sum(inter_class_similarity_net_exp * att_mask, dim=1), min=1e-30))
        )

        nll = inter_class_loss1 = torch.sum(inter_class_loss1) / torch.sum(spec_lengths)

        inter_class_similarity_max2, _ = torch.max(logit_proj, dim=2, keepdim=True)
        inter_class_similarity2_net = logit_proj - inter_class_similarity_max2
        inter_class_similarity2_net_exp = torch.exp(inter_class_similarity2_net)
        inter_class_loss2 = text_mask * (
            - torch.log(torch.clamp(torch.sum(inter_class_similarity2_net_exp * match_mask, dim=2), min=1e-30)) \
            + torch.log(torch.clamp(torch.sum(inter_class_similarity2_net_exp * att_mask, dim=2), min=1e-30))
        )
        inter_class_loss2 = torch.sum(inter_class_loss2) / torch.sum(text_lengths)
        att_loss += inter_class_loss1 + 0.1 * inter_class_loss2
        
        with torch.no_grad():
            # Attention sharpness
            att_soft = inter_class_similarity_net_exp * att_mask
            att_soft = F.normalize(att_soft, 1, 1)                      # N x T_enc x T_dec
            sharpness = torch.mean(1 - torch.max(att_soft, dim=1)[0])

            # Upsample attention if needed
            if self.att_upsample and self.r_factor != 4:
                attention = F.interpolate(attention, scale_factor=4//self.r_factor)
                att_mask = F.interpolate(att_mask.float(), scale_factor=4//self.r_factor).long()
        
        return attention, att_loss, att_mask, sharpness, nll

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


class ARAttention(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, att_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, dropout_p=0.5, debug=0):
        super(ARAttention, self).__init__()
        self.O_dec = output_size

        def bias_layer(in_dim, out_dim, bias=True):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=bias),
                nn.Softsign()
            )

        # outputs of the following layers are reusable through recurrence
        self.in_att_linear_enc = bias_layer(enc_hidden, att_hidden, bias=True)
        self.in_att_linear_spkr = bias_layer(spkr_embed_size, att_hidden, bias=False)
        self.in_att_conv_prev_att = nn.Conv1d(1, att_hidden, 31, padding=15, bias=False)

        self.in_att_linear_dec = nn.Linear(dec_hidden, att_hidden, bias=False)
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
        
        self.set_attention_range(att_range)

    def forward(self, input_enc, input_dec, prev_attention, spkr_vec, lengths_enc, speed, debug=0, **kwargs):
        N, T_enc = input_enc.size(0), lengths_enc.max().item()

        input_dec = torch.cat([input_dec, spkr_vec], dim=-1)
        out_prenet = self.prenet(input_dec)  # N x O_dec -> N x 1 x H

        if self.null_bias:
            # reusable bias terms
            if speed is None:
                self.att_bias_speed = 0
            else:
                self.att_bias_speed = self.in_att_speed(speed.unsqueeze(-1)).unsqueeze(1)
            self.att_bias_enc = self.in_att_linear_enc(input_enc)                           # N x T_enc x H_att
            self.att_bias_spkr = self.in_att_linear_spkr(spkr_vec).expand_as(self.att_bias_enc)
            self.null_bias = False

        # attention -- https://arxiv.org/pdf/1506.07503.pdf
        in_att_prev_att = self.in_att_conv_prev_att(prev_attention.transpose(1, 2)).transpose(1, 2)
        in_att_dec = self.in_att_linear_dec(out_prenet)
        e = self.att_bias_enc + in_att_dec + self.att_bias_spkr + in_att_prev_att + self.att_bias_speed     # N x T_enc x H_att

        # attention mask (confine attention to be formed near previously attended characters)
        with torch.no_grad():
            att_mask = prev_attention.data.new().resize_(N, T_enc).zero_()
            _, att_max_idx = torch.max(prev_attention.data, dim=1)
            for i in range(self.att_range):
                idx1 = torch.min(
                    torch.clamp((att_max_idx + i), min=0), 
                    lengths_enc.sub(1).type_as(att_max_idx)
                ).long()
                idx2 = torch.min(
                    torch.clamp((att_max_idx - i), min=0),
                    lengths_enc.sub(1).type_as(att_max_idx)
                ).long()
                att_mask.scatter_(1, idx1, 1)
                att_mask.scatter_(1, idx2, 1)
            att_mask = att_mask.view(N, T_enc, 1)

        # stable softmax
        logit = self.att_proj(torch.tanh(e))
        logit_max, _ = torch.max(logit, dim=1, keepdim=True)
        curr_attention = torch.exp(logit - logit_max) * att_mask
        curr_attention = F.normalize(curr_attention, 1, 1)                      # N x T_enc x 1
        return curr_attention

    def set_attention_range(self, r):
        self.att_range = r

    def reset_bias(self):
        # need to reset bias at every iteration to avoid unnecessary computation
        self.att_bias_enc = None
        self.att_bias_spkr = None
        self.att_bias_speed = None

        self.null_bias = True


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, spkr_embed_size=-1, debug=0, **kwargs):
        super(DurationPredictor, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # speaker/speed biases will be sliced.
        self.speaker_proj = nn.Linear(spkr_embed_size, filter_channels * 2)
        self.speed_proj = nn.Linear(1, filter_channels * 2)

    def forward(self, x, x_mask, spkr_vec=None, speed=None, debug=0, **kwargs):
        N = x.size(0)
        spkr_biases = self.speaker_proj(spkr_vec).view(N, -1, 1)    # N x 2H x 1
        speed_biases = self.speed_proj(speed.unsqueeze(1)).unsqueeze(2)
        spkr_bias1, spkr_bias2 = torch.chunk(spkr_biases, 2, dim=1)
        speed_bias1, speed_bias2 = torch.chunk(speed_biases, 2, dim=1)
        global_bias1 = spkr_bias1 + speed_bias1
        global_bias2 = spkr_bias2 + speed_bias2

        x = self.conv_1(x * x_mask) + global_bias1
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask) + global_bias2
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return (x * x_mask).squeeze(1)


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps

      self.gamma = nn.Parameter(torch.ones(channels))
      self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x


def get_attention_mask(text_lengths, whole_spec_len, device=None):
    N = text_lengths.size(0)
    T_enc = text_lengths.max().item()
    T_dec = whole_spec_len.max().item()

    step_idx = torch.arange(1, max(T_enc, T_dec)+1, device=device)
    enc_step_idx = step_idx[:T_enc].view(1, T_enc, 1).expand(N, -1, -1)
    dec_step_idx = step_idx[:T_dec].view(1, 1, T_dec).expand(N, -1, -1)

    enc_mask = torch.le(enc_step_idx, text_lengths.view(N, 1, 1).expand(-1, T_enc, -1))
    dec_mask = torch.le(dec_step_idx, whole_spec_len.view(N, 1, 1).expand(-1, -1, T_dec))
    return enc_mask * dec_mask

