import torch
import torch.nn as nn
import torch.nn.functional as F

from tacotron.module.commons import maximum_path


class Attention(nn.Module):
    def __init__(self, vocab_size, mel_dim, att_hidden=128, kernel_size=3, debug=0, **kwargs):
        super(Attention, self).__init__()
        self.vocab_size = vocab_size
        aux_kernel_size = 3
        num_spec_enc_layers = 6
        spec_enc_hidden = 2 * att_hidden

        # text encoder
        self.embedding = nn.Embedding(vocab_size, att_hidden)

        if kernel_size == 3:
            padding_size = 1
        elif kernel_size == 1:
            padding_size = 0
        else:
            raise RuntimeError(f'kernel_size {kernel_size} is not supported.')

        # main speech encoder
        in_channel = mel_dim
        self.conv_ref_enc = nn.ModuleList()
        self.out_channel_ref = [spec_enc_hidden for _ in range(num_spec_enc_layers)]
        for i, c in enumerate(self.out_channel_ref):
            if i % 2 == 0:
                self.conv_ref_enc.append(
                    nn.Sequential(
                        nn.Conv1d(in_channel, c, 3, 1, 1, bias=False),
                        nn.BatchNorm1d(c),
                    )
                )
            else:
                self.conv_ref_enc.append(
                    nn.Sequential(
                        nn.Conv1d(in_channel, c, kernel_size, 1, padding_size, bias=False),
                        nn.BatchNorm1d(c),
                    )
                )
            in_channel = c

        # auxiliary speech encoder
        in_channel = mel_dim
        self.conv_ref_enc_aux = nn.ModuleList()
        self.out_channel_ref_aux = [spec_enc_hidden for _ in range(num_spec_enc_layers)]
        for i, c in enumerate(self.out_channel_ref_aux):
            if i % 2 == 0:
                self.conv_ref_enc_aux.append(
                    nn.Sequential(
                        nn.Conv1d(in_channel, c, 3, 1, 1, bias=False),
                        nn.BatchNorm1d(c),
                    )
                )
            else:
                self.conv_ref_enc_aux.append(
                    nn.Sequential(
                        nn.Conv1d(in_channel, c, kernel_size, 1, padding_size, bias=False),
                        nn.BatchNorm1d(c),
                    )
                )
            in_channel = c

        self.query_proj = nn.Linear(self.out_channel_ref[-1], att_hidden)
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

        # BN debugging
        if N == 1:
            self_is_train = self.training
            self.eval()

        # from text
        text_mask = kwargs.get('text_mask')
        key = self.embedding(text) * text_mask.unsqueeze(2)

        # from spectrogram
        enc_spec_mask = torch.arange(0, T_dec, device=spec.device).view(1, -1).expand(N, -1)
        enc_spec_mask = torch.lt(enc_spec_mask, spec_lengths.view(-1, 1).expand(-1, T_dec))                 # N x T_dec
        enc_spec_mask = enc_spec_mask.view(N, 1, T_dec)
        output_ref_enc = output_ref_enc_aux = spec.transpose(1, 2)          # N x C x T

        # main encoder
        residual = 0
        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = output_ref_enc * enc_spec_mask
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)
            if i % 2 == 1:
                output_ref_enc = residual = output_ref_enc + residual
            if i < len(self.conv_ref_enc)-1:
                output_ref_enc = F.relu(output_ref_enc)
        output = output_ref_enc.transpose(1, 2)                             # N x T x C

        key = key / torch.clamp(torch.norm(key, p=2, dim=2, keepdim=True), min=1e-8)
        query = self.query_proj(output.contiguous())                        # N x T_dec x H_att
        query = query / torch.clamp(torch.norm(query, p=2, dim=2, keepdim=True), min=1e-8)
        cos_similarity = torch.bmm(key, query.transpose(1, 2))                  # N x T_enc x T_dec
        short_token_mask = kwargs.get('short_token_mask')       # N x T_enc x 1
        cos_similarity = (1-short_token_mask) * cos_similarity - short_token_mask
        similarity = 10 * torch.exp(self.sim_proj_w) * cos_similarity + self.sim_proj_b

        # aux encoder
        residual = 0
        for i in range(len(self.conv_ref_enc_aux)):
            output_ref_enc_aux = output_ref_enc_aux * enc_spec_mask
            output_ref_enc_aux = self.conv_ref_enc_aux[i](output_ref_enc_aux)
            if i % 2 == 1:
                output_ref_enc_aux = residual = output_ref_enc_aux + residual
            if i < len(self.conv_ref_enc_aux)-1:
                output_ref_enc_aux = F.relu(output_ref_enc_aux)
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
        
        # silence promotion
        lmda2 = 0.01
        silence_mask = text_mask.long()
        silence_mask[:, :-1] -= text_mask[:, 1:].long()
        silence_mask[:, 0].fill_(1)
        silence_mask = silence_mask.unsqueeze(2)
        
        energy = torch.mean(torch.exp(spec[:, :, 20:]), dim=2).unsqueeze(1)
        silence_energy_max = torch.sum(energy * silence_mask * attention, dim=1)
        silence_energy_max, _ = torch.max(silence_energy_max, dim=1)
        silence_energy_max = silence_energy_max.view(N, 1, 1).expand_as(attention)
        non_silence_energy_min = torch.sum(energy * (1-silence_mask) * attention, dim=1)
        non_silence_energy_min = torch.eq(non_silence_energy_min, 0).type_as(energy) * 100 + non_silence_energy_min
        non_silence_energy_min, _ = torch.min(non_silence_energy_min, dim=1)
        non_silence_energy_min = non_silence_energy_min.view(N, 1, 1).expand_as(attention)
        decision_boundary = (silence_energy_max + non_silence_energy_min) / 2
        promotion_mask = torch.le(energy, decision_boundary) * silence_mask
        silence_promotion = - lmda2 * torch.sum(promotion_mask * att_mask * logsig_similarity, dim=(1, 2)) \
            / torch.clamp(torch.sum(promotion_mask * att_mask, dim=(1, 2)), min=1)
        att_loss += torch.mean(silence_promotion)

        # BN debugging
        if N == 1:
            if self_is_train:
                self.train()
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
