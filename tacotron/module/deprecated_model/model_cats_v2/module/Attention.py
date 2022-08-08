import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tacotron.module.deprecated_model.model_cats_v2.module.commons import maximum_path


class Attention(nn.Module):
    def __init__(self, enc_hidden, mel_dim, att_hidden=128, num_spkr=1, debug=0, **kwargs):
        super(Attention, self).__init__()
        self.vocab_size = kwargs.get('vocab_size')
        aux_kernel_size = 3
        spkr_hidden = 64
        num_spec_enc_layers = 6
        spec_enc_hidden = 2 * att_hidden


        self.spkr_emb = nn.Embedding(num_spkr, spkr_hidden)

        # text encoder
        self.enc_proj = nn.Linear(enc_hidden, att_hidden, bias=True)

        # main speech encoder
        in_channel = mel_dim
        self.conv_ref_enc = nn.ModuleList()
        self.spkr_biases = nn.ModuleList()
        self.gst_biases = nn.ModuleList()
        self.out_channel_ref = [spec_enc_hidden for _ in range(num_spec_enc_layers)]
        for c in self.out_channel_ref:
            self.conv_ref_enc.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, c, 3, 1, 1),
                    nn.GroupNorm(c, c),
                    nn.ReLU()
                )
            )
            self.spkr_biases.append(
                nn.Sequential(
                    nn.Linear(spkr_hidden, spkr_hidden),
                    nn.ReLU(),
                    nn.Linear(spkr_hidden, c),
                    nn.ReLU()
                )
            )
            self.gst_biases.append(
                nn.Sequential(
                    nn.Linear(enc_hidden, spkr_hidden),
                    nn.ReLU(),
                    nn.Linear(spkr_hidden, c),
                    nn.ReLU()
                )
            )
            in_channel = c

        # auxiliary speech encoder
        in_channel = mel_dim
        self.conv_ref_enc_aux = nn.ModuleList()
        self.spkr_biases_aux = nn.ModuleList()
        self.gst_biases_aux = nn.ModuleList()
        self.out_channel_ref_aux = [spec_enc_hidden for _ in range(num_spec_enc_layers)]
        for c in self.out_channel_ref_aux:
            self.conv_ref_enc_aux.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, c, 3, 1, 1),
                    nn.GroupNorm(c, c),
                    nn.ReLU()
                )
            )
            self.spkr_biases_aux.append(
                nn.Sequential(
                    nn.Linear(spkr_hidden, spkr_hidden),
                    nn.ReLU(),
                    nn.Linear(spkr_hidden, c),
                    nn.ReLU()
                )
            )
            self.gst_biases_aux.append(
                nn.Sequential(
                    nn.Linear(enc_hidden, spkr_hidden),
                    nn.ReLU(),
                    nn.Linear(spkr_hidden, c),
                    nn.ReLU()
                )
            )
            in_channel = c

        self.query_proj = nn.Linear(self.out_channel_ref[-1], att_hidden-1)
        self.ctc_proj = nn.Linear(self.out_channel_ref_aux[-1], self.vocab_size)

        self.register_parameter('sim_proj_w', nn.Parameter(torch.tensor([[[0.0]]])))
        self.register_parameter('sim_proj_b', nn.Parameter(torch.tensor([[[-5.0]]])))

        self.aux_pad_size = (aux_kernel_size - 1) // 2
        self.aux_kernel = torch.ones(1, 1, aux_kernel_size, 1)

    def forward(self, text, spec, text_lengths, spec_lengths, debug=0, **kwargs):
        """ spec: N x T x O_dec sized Tensor (Spectrogram)
            seq_style_vec: N x T x H sized Tensor
        """
        N, T_dec, _ = spec.size()            
        spkr_vec = kwargs.get('spkr_vec')
        gst_vec = kwargs.get('gst_vec')

        # from text
        key = self.enc_proj(text)                               # N x T_enc x H_att

        # from spectrogram
        enc_spec_mask = torch.arange(0, T_dec, device=spec.device).view(1, -1).expand(N, -1)
        enc_spec_mask = torch.lt(enc_spec_mask, spec_lengths.view(-1, 1).expand(-1, T_dec))                 # N x T_dec
        enc_spec_mask = enc_spec_mask.view(N, 1, T_dec)
        output_ref_enc = output_ref_enc_aux = spec.transpose(1, 2)          # N x C x T

        # main encoder
        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = output_ref_enc * enc_spec_mask
            if i % 2 == 1:
                output_ref_enc = output_ref_enc \
                    + self.conv_ref_enc[i](output_ref_enc) \
                    + self.spkr_biases[i](spkr_vec).view(N, -1, 1) \
                    + self.gst_biases[i](gst_vec).view(N, -1, 1)
            else:
                output_ref_enc = self.conv_ref_enc[i](output_ref_enc) \
                    + self.spkr_biases[i](spkr_vec).view(N, -1, 1) \
                    + self.gst_biases[i](gst_vec).view(N, -1, 1)
        output = output_ref_enc.transpose(1, 2)                             # N x T x C

        key = key / torch.clamp(torch.norm(key, p=2, dim=2, keepdim=True), min=1e-8)
        query = self.query_proj(output.contiguous())                        # N x T_dec x H_att
        energy = torch.mean(spec, dim=2, keepdim=True)
        query = torch.cat([query, energy], dim=2)
        query = query / torch.clamp(torch.norm(query, p=2, dim=2, keepdim=True), min=1e-8)
        similarity = torch.bmm(key, query.transpose(1, 2))                  # N x T_enc x T_dec
        similarity = 10 * torch.exp(self.sim_proj_w) * similarity + self.sim_proj_b

        # aux encoder
        for i in range(len(self.conv_ref_enc_aux)):
            output_ref_enc_aux = output_ref_enc_aux * enc_spec_mask
            if i % 2 == 1:
                output_ref_enc_aux = output_ref_enc_aux \
                    + self.conv_ref_enc_aux[i](output_ref_enc_aux) \
                    + self.spkr_biases_aux[i](spkr_vec).view(N, -1, 1) \
                    + self.gst_biases_aux[i](gst_vec).view(N, -1, 1)
            else:
                output_ref_enc_aux = self.conv_ref_enc_aux[i](output_ref_enc_aux) \
                    + self.spkr_biases_aux[i](spkr_vec).view(N, -1, 1) \
                    + self.gst_biases_aux[i](gst_vec).view(N, -1, 1)
        output_aux = output_ref_enc_aux.transpose(1, 2)                             # N x T x C
        
        ctc_key = kwargs.get('enc_input')
        ctc_key = F.one_hot(ctc_key, num_classes=self.vocab_size).float()
        ctc_query_logit = self.ctc_proj(output_aux.contiguous())                        # N x T_dec x V
        ctc_query = F.softmax(ctc_query_logit, dim=2)
        similarity_ctc = torch.bmm(ctc_key, ctc_query.transpose(1, 2))                  # N x T_enc x T_dec

        with torch.no_grad():
            # main mask
            text_mask = torch.arange(0, key.size(1), device=key.device).view(1, -1).expand(N, -1)
            text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, key.size(1)))               # N x T_enc
            spec_mask = torch.arange(0, query.size(1), device=key.device).view(1, -1).expand(N, -1)
            spec_mask = torch.lt(spec_mask, spec_lengths.view(-1, 1).expand(-1, query.size(1)))             # N x T_dec
            att_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(1)                                      # N x T_enc x T_dec

            # main attention
            lsmx = F.logsigmoid(similarity) * att_mask
            lsmx_att = lsmx - torch.eq(lsmx, 0).float() * lsmx.min()
            match_mask = maximum_path(lsmx_att, att_mask).detach()
            attention = match_mask

            # aux attention
            lsmx = F.logsigmoid(similarity_ctc) * att_mask
            lsmx_aux = lsmx - torch.eq(lsmx, 0).float() * lsmx.min()
            attention_aux = maximum_path(lsmx_aux, att_mask).detach()

            self.aux_kernel = self.aux_kernel.to(text.device)
            attention_aux = F.conv2d(
                attention_aux.unsqueeze(1),
                self.aux_kernel,
                padding=(self.aux_pad_size, 0)
            )
            attention_aux = attention_aux.squeeze(1) * att_mask

        att_loss = 0
        # NCE loss for main attention
        logsig_similarity = F.logsigmoid(similarity)
        neg_logsig_similarity = F.logsigmoid(-similarity)

        inter_class_loss = - (
            match_mask * logsig_similarity + (1-match_mask) * att_mask * neg_logsig_similarity
        )
        inter_class_loss = torch.sum(inter_class_loss, dim=(1, 2)) / torch.sum(att_mask, dim=(1, 2))
        nll = torch.mean(inter_class_loss)
        att_loss += nll

        # regularization with aux attention
        lmda = 0.5
        att_aux_loss = - (
            attention_aux * logsig_similarity + (1-attention_aux) * att_mask * neg_logsig_similarity
        )
        att_aux_loss = torch.sum(att_aux_loss, dim=(1, 2)) / torch.sum(att_mask, dim=(1, 2)) * lmda
        att_loss += torch.mean(att_aux_loss)

        # ctc loss for aux attention
        phoneme_target = kwargs.get('enc_input')
        ctc_input = ctc_query_logit.transpose(0, 1)        # T_dec x N x V
        ctc_input = F.log_softmax(ctc_input, dim=2)
        ctc_loss = F.ctc_loss(ctc_input, phoneme_target, spec_lengths, text_lengths)
        att_loss += ctc_loss
        return attention, att_loss, att_mask, nll

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
