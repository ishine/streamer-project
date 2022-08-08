# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

class Tacotron(nn.Module):
    def __init__(self, args, **kwargs):
        super(Tacotron, self).__init__()
        self.trunc_size = args.trunc_size
        self.r_factor = args.r_factor
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.fluency = args.fluency
        self.att_weights = None
        self.prev_dec_output = None
        self.detach = True

        if args.debug == 8:
            self.encoder = Encoder2(args.vocab_size, args.charvec_dim, args.enc_hidden, args.dropout, args.spkr_embed_size, fluency=args.fluency, debug=args.debug)
        else:
            if args.conv == 0:
                self.encoder = Encoder(args.vocab_size, args.charvec_dim, args.enc_hidden, args.dropout, args.spkr_embed_size, fluency=args.fluency, debug=args.debug)
            else:
                self.encoder = EncoderConv(args.vocab_size, args.charvec_dim, args.enc_hidden, args.spkr_embed_size, debug=args.debug)

        self.spkr_embed = nn.Embedding(args.num_id, args.spkr_embed_size, max_norm=2)       # randomly chosen norm size

        if args.debug == 10 or args.debug == 11:
            self.text_spkr_classifier = nn.Sequential(
                nn.Linear(args.enc_hidden, args.dec_hidden),
                nn.Tanh(),
                nn.Linear(args.dec_hidden, args.num_id),
            )

        if args.debug == 2:
            self.ref_encoder = ReferenceEncoder(args.dec_out_size, args.spkr_embed_size, args.att_hidden, args.n_token, debug=args.debug, n_head=args.n_head)
            self.prosody_stats = ProsodyStatsGST(args.num_id, args.spkr_embed_size, debug=args.debug)
        else:
            if args.exp_no in ['170', '200', '202', 'yk8', 'yk9']:
                self.ref_encoder = ReferenceEncoderOld(args.dec_out_size, args.enc_hidden, args.att_hidden, args.n_token, debug=args.debug, n_head=args.n_head)
            else:
                self.ref_encoder = ReferenceEncoder(args.dec_out_size, args.enc_hidden, args.att_hidden, args.n_token, debug=args.debug, n_head=args.n_head, spkr_embed_size=args.spkr_embed_size)
            self.prosody_stats = ProsodyStatsGST(args.num_id, args.enc_hidden, debug=args.debug)

        if args.debug == 1 or args.debug == 13 or args.debug == 7 or args.debug == 15 or args.debug == 9:
            self.decoder = AttnDecoderRNN2(args.enc_hidden, args.att_hidden, args.dec_hidden, args.dec_out_size, args.spkr_embed_size, args.att_range, args.r_factor, args.dropout, debug=args.debug)
        elif args.debug == 12 or args.debug == 14 or args.debug == 16:
            self.decoder = AttnDecoderRNN3(args.enc_hidden, args.att_hidden, args.dec_hidden, args.dec_out_size, args.spkr_embed_size, args.att_range, args.r_factor, args.dropout, debug=args.debug, num_t_layer=args.num_trans_layer)
        else:
            self.decoder = AttnDecoderRNN(args.enc_hidden, args.att_hidden, args.dec_hidden, args.dec_out_size, args.spkr_embed_size, args.att_range, args.r_factor, args.dropout, debug=args.debug)

        if args.dec_out_type == 'mel':
            self.post_processor = NewPostProcessor(args.dec_hidden * 2, args.dec_out_size, args.post_out_size, args.dropout)
        else:
            self.post_processor = PostProcessor(args.att_hidden, args.dec_out_size, args.post_out_size, 8)

        # GST sparsity constraint
        self.sp_reg_weight = 0.1
        self.sp_param = 0.5
        self.sp_constant = self.sp_param * np.log(self.sp_param) + (1 - self.sp_param) * np.log(1 - self.sp_param)
        self.sp_constant *= args.n_token

        # attention sharpness to measure generation quality
        self.attention_sharpness = 0

    def forward(self, enc_input, dec_input, spkr_id, spec_lengths, text_lengths, spkr_vec=None, debug=0, gst_vec=None, gst_source=None, stop_type=0, speed_x=1.0, **kwargs):
        N, r = enc_input.size(0), self.r_factor
        T_enc = enc_input.size(1)
        T_wav, T_dec = min(spec_lengths), min(spec_lengths)//r
        output_mel_list = []
        stop_token_list = []
        
        # consider compatibility
        style_vec = gst_vec

        if spkr_vec is None:
            spkr_vec = self.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        else:
            spkr_vec = spkr_vec

        # set speed
        if debug == 1 or debug == 13 or debug == 12 or debug == 14 or debug == 7 or debug == 15 or debug == 16 or debug == 9:
            if kwargs.get('speed') is not None:
                speed = spkr_vec.new().resize_(N).fill_(kwargs.get('speed'))
            else:
                if gst_source == 'ref_wav' or self.training:
                    speed = torch.Tensor(text_lengths).type_as(spkr_vec) / torch.Tensor(spec_lengths).type_as(spkr_vec)  # N
                else:
                    speed = torch.index_select(self.prosody_stats.speed, 0, spkr_id).squeeze(1)
                speed /= speed_x

        if debug == 1 and self.training:
            if random.random() > 0.5:
                dec_input = F.avg_pool2d(dec_input, 3, 1, 1)

        # get style vector
        if gst_source == 'ref_wav' or self.training:
            style_vec, gst_att = self.ref_encoder(dec_input, spkr_vec=spkr_vec, debug=debug)  # N x 1 x style_dim
        elif gst_source == 'cluster':
            # assert (N == 1)
            # Use curated style token
            if N == 1:
                style_vec = style_vec.view(1, 1, -1)
            else:
                style_vec = style_vec.view(N, 1, -1)
        elif gst_source == 'gst_mean':
            # assert (N == 1)
            # Use mean style token saved before. (Usually in evaluation)
            style_vec_question = torch.index_select(self.prosody_stats.question, 0, spkr_id).unsqueeze(1)   # N x 1 x style_dim
            style_vec_mean = torch.index_select(self.prosody_stats.means, 0, spkr_id).unsqueeze(1)          # N x 1 x style_dim

            mask_question = [1 if 134 in enc_input[i].tolist() else 0 for i in range(enc_input.size(0))]    # 134 is "?"
            mask_question = torch.Tensor(mask_question).type_as(style_vec_question).view(N, 1, 1)
            style_vec = mask_question * style_vec_question + (1 - mask_question) * style_vec_mean
        else:
            raise RuntimeError(f'Not supported style source: {gst_source}')

        if debug == 6:
            style_vec = self.prosody_stats.normalize_prosody(style_vec, spkr_id)

        if debug == 2:
            spkr_vec = style_vec

        if self.fluency == 1:
            enc_output = self.encoder(enc_input, text_lengths, spkr_vec=None, debug=debug)
        else:
            enc_output = self.encoder(enc_input, text_lengths, spkr_vec=spkr_vec, debug=debug)

        if debug == 10 or debug == 11:
            rev_enc_output = enc_output

        if debug == 2:
            pass
        else:
            enc_output = enc_output + style_vec

        # Generation length FIX
        sequence_end = 0
        if not self.training:
            if stop_type == 0:
                is_att_ended = spkr_vec.data.new().resize_(N).long().zero_()
                is_volume_ended = is_att_ended.data.new().resize_(N).zero_()
                max_mel_norm = spkr_vec.data.new().resize_(N).zero_()
                min_mel_norm = spkr_vec.data.new().resize_(N).fill_(1000)
                latest_max_position = is_att_ended.data.new().resize_(N).fill_(-1)
                end_condition = is_att_ended.data.new().resize_(N).zero_()
                text_end_position = torch.clamp(torch.Tensor(text_lengths).type_as(is_att_ended.data) - 1, min=0)
                expected_seq_end = torch.clamp(text_end_position, min=4)
                sequence_end = is_att_ended.data.new().resize_(N).zero_()
                att_offset = is_att_ended.data.new().resize_(N).zero_()

                # calibrate by speech speed
                speed_calib = torch.clamp(torch.ceil(1 / speed / self.r_factor) - 1, min=0)
            elif stop_type == -1:
                is_att_ended = torch.Tensor(text_lengths).type_as(enc_input.data).fill_(0)
                is_volume_ended = torch.Tensor(text_lengths).type_as(enc_input.data).fill_(0)
                max_mel_norm = torch.Tensor(text_lengths).type_as(enc_output.data).fill_(0)
                min_mel_norm = torch.Tensor(text_lengths).type_as(enc_output.data).fill_(1000)
                end_condition = torch.Tensor(text_lengths).type_as(enc_input.data).fill_(0)
                text_end_position = torch.clamp(torch.Tensor(text_lengths).type_as(enc_input.data) - 2, min=0)
                sequence_end = torch.Tensor([0 for _ in range(N)]).type_as(enc_input.data)
            elif stop_type == 1:
                end_buffer = 5
                text_end_position = torch.Tensor(text_lengths).type_as(spkr_vec.data).long()
                sequence_end = torch.Tensor([end_buffer for _ in range(N)]).type_as(spkr_vec.data).long()
        # Generation length FIX

        self.decoder.reset_bias()
        for di in range(T_dec):
            end_idx = (di+1)*r - 1

            if debug == 1 or debug == 12 or debug == 14 or debug == 7 or debug == 15 or debug == 16 or debug == 9:
                self.prev_dec_output, context_target = self.decoder(enc_output, self.prev_dec_output, spkr_vec, text_lengths, speed, debug=debug, context_vec=None)
            elif debug == 13:
                curr_dec_output = dec_input[:, end_idx-(r-1):end_idx+1].view(N, -1).contiguous()
                self.prev_dec_output, context_target = self.decoder(enc_output, self.prev_dec_output, spkr_vec, text_lengths, speed, debug=debug, context_vec=None, output_dec=curr_dec_output)
            else:
                self.prev_dec_output, context_target = self.decoder(enc_output, self.prev_dec_output, spkr_vec, text_lengths, debug=debug, context_vec=None)
            output_mel_list.append(self.prev_dec_output)
            stop_token_list.append(context_target)

            # compute attention sharpness
            curr_max_weights, curr_max_position = torch.max(self.decoder.att_weights.data.view(N, -1), dim=1)
            self.attention_sharpness = self.attention_sharpness - torch.sum(curr_max_weights - 1)

            if not self.training:
                self.att_weights.append(self.decoder.att_weights.data)

                # Generation length FIX
                if stop_type == 0:
                    # Denormalize & Convert back to linear
                    spec = (torch.clamp(self.prev_dec_output, 0, 1) * 100) - 100
                    spec = torch.pow(10.0, spec * 0.05)

                    # Stopping criteria
                    curr_mel_norm, _ = torch.topk(torch.norm(spec, dim=2), min(2, self.r_factor), dim=1, largest=True)
                    curr_mel_norm = curr_mel_norm[:, -1]
                    buffer = torch.clamp((max_mel_norm - min_mel_norm) * 0.03, min=0)
                    if di > 4:  # some speakers may have silence for 3 time-steps
                        sequence_end = end_condition * sequence_end + (1 - end_condition) * di

                        is_att_ended = torch.le(expected_seq_end, di).long()
                        is_volume_ended = torch.max(is_volume_ended,
                                                    torch.le(curr_mel_norm, min_mel_norm + buffer).long() * is_att_ended)
                        end_condition = torch.max(end_condition, is_volume_ended)

                    # compute offset of attention
                    att_offset = att_offset \
                                 + (torch.eq(att_offset, 0) * torch.ge(curr_mel_norm, min_mel_norm * 10)).long() * int(di * 0.5)

                    min_mel_norm = torch.min(min_mel_norm, curr_mel_norm)
                    max_mel_norm = torch.max(max_mel_norm, curr_mel_norm)

                    update_mask = torch.gt(curr_max_position, latest_max_position).long()
                    latest_max_position = torch.min(torch.max(curr_max_position, latest_max_position), latest_max_position + 2)

                    new_expected_seq_end = (text_end_position.float() / (latest_max_position + 1).float()) * (di+1) \
                                           + speed_calib \
                                           + att_offset.float()
                    expected_seq_end = update_mask * torch.clamp(torch.ceil(new_expected_seq_end).long(), min=5) \
                                       + (1 - update_mask) * expected_seq_end

                    if end_condition.prod().item() == 1:
                        break
                elif stop_type == -1:
                    # old-style stopping (compute norm from log-mel)
                    curr_mel_norm = torch.norm(self.prev_dec_output[:, -1], dim=1)
                    if di > 0:
                        buffer = (max_mel_norm - min_mel_norm) / 25
                        is_att_ended = torch.max(is_att_ended, torch.ge(curr_max_position, text_end_position).long())
                        is_volume_ended = torch.max(is_volume_ended, torch.le(curr_mel_norm, min_mel_norm + buffer).long() * is_att_ended)
                        is_norm_rebounded = torch.gt(curr_mel_norm, prev_mel_norm).long()
                        end_condition = torch.max(end_condition, is_volume_ended * is_norm_rebounded)
                        sequence_end = end_condition * sequence_end + (1 - end_condition) * di

                    min_mel_norm = torch.min(min_mel_norm, curr_mel_norm)
                    max_mel_norm = torch.max(max_mel_norm, curr_mel_norm)
                    prev_mel_norm = curr_mel_norm
                    if end_condition.prod().item() == 1:
                        break
                elif stop_type == 1:
                    isNotEnded = torch.lt(curr_max_position, text_end_position - 2)
                    condition = (isNotEnded).long()
                    sequence_end = (1 - condition) * sequence_end + condition * (di + end_buffer)

                    if N == 1 and sequence_end.squeeze().item() == di:
                        break
                # elif stop_type == 2:
                #     # use network to predict stop token
                #     output_stop = torch.sigmoid(context_target)                         # N x 1 x 1
                #     stop_threshold = 0.6
                #     if N == 1 and output_stop.squeeze().item() > stop_threshold:
                #         break
                # Generation length FIX

            if random.random() < self.teacher_forcing_ratio:
                self.prev_dec_output = dec_input[:, end_idx]                       # Teacher forcing
            else:
                self.prev_dec_output = self.prev_dec_output[:, -1]


        output_dec = torch.cat(output_mel_list, dim=1)
        output_post = self.post_processor(output_dec)

        if debug == 10 or debug == 11:
            if self.detach:
                rev_enc_output = rev_enc_output.detach()

            if debug == 10:
                lamb = 0.01
            elif debug == 11:
                lamb = 0.001

            rev_enc_output = grad_reverse(rev_enc_output.contiguous().view(N * T_enc, -1), lamb, 0.5)
            speaker_classification_logit = self.text_spkr_classifier(rev_enc_output)            # NT x H
            spkr_target = spkr_id.view(N, 1).repeat(1, T_enc).view(-1)
            spkr_adv_loss = F.cross_entropy(speaker_classification_logit, spkr_target)
        else:
            spkr_adv_loss = 0

        if (debug == 15 or debug == 16) and self.training:
            pred_prenet = self.decoder.prenet(torch.cat([output_dec[:, :T_wav], spkr_vec.expand(-1, T_wav, -1)], dim=-1))
            gt_prenet = self.decoder.prenet(torch.cat([dec_input[:, :T_wav], spkr_vec.expand(-1, T_wav, -1)], dim=-1))
            prenet_loss = F.mse_loss(pred_prenet, gt_prenet.detach()) * 10
        else:
            prenet_loss = 0

        # compute attention sharpness to measure generation quality.
        # Note that the error is not backpropagated from attention sharpness.
        self.attention_sharpness = self.attention_sharpness.data.item() / (N * T_dec)

        return {
            "output_dec": output_dec,
            "output_post": output_post,
            "style_vec": style_vec,
            "prenet_loss": prenet_loss,
            "spkr_adv_loss": spkr_adv_loss,
            "seq_end": sequence_end * r
        }

    def reset_decoder_states(self, debug=0):
        # wrapper
        self.att_weights = []
        self.decoder.reset_states(debug=debug)
        self.prev_dec_output = None

    def mask_decoder_states(self, len_mask, debug=0):
        # wrapper
        self.decoder.mask_states(len_mask, debug=debug)

        if len_mask is None:
            if self.prev_dec_output is not None:
                self.prev_dec_output = self.prev_dec_output.data
        else:
            if self.prev_dec_output is not None:
                self.prev_dec_output = torch.index_select(self.prev_dec_output.data, 0, len_mask).data

    def freeze_params(self, module_list):
        def freeze_params(nn_module):
            for param in nn_module.parameters():
                param.requires_grad = False

        l = set(module_list)
        if 's' in l:
            print('freeze speaker embedding.')
            freeze_params(self.spkr_embed)
        elif 'e' in l:
            print('freeze phoneme embedding.')
            freeze_params(self.encoder.embedding)

    def import_speaker_embedding_matrix(self, speaker_embedding_matrix):
        self.spkr_embed.weight = speaker_embedding_matrix.type_as(self.spkr_embed.weight.data)

    def export_speaker_embedding_matrix(self):
        return self.spkr_embed.weight

    def get_mixed_speaker_vector(self, speaker_id_list, weight_list):
        """
        Use this to mix speaker embeddings with specific weight
        :param speaker_id_list: (List) List of N speaker id (compact id from the speaker_manager)
        :param weight_list: (List) weight to mix
        :return: (Variable) speaker vector
        """
        assert len(speaker_id_list) == len(weight_list)
        spkr_id_tensor = torch.LongTensor(speaker_id_list)            # N
        weight_tensor = torch.Tensor(weight_list)                     # N

        spkr_vecs = self.spkr_embed(spkr_id_tensor)                             # N x S
        result = torch.matmul(spkr_vecs.t(), weight_tensor).view(1,1,-1)        # 1 x 1 x S
        return result

    def get_velocity_loss(self):
        return self.decoder.get_velocity_loss()

    def set_attention_range(self, att_range):
        self.decoder.set_attention_range(att_range)


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
        output = self.conv_1st(self.embedding(input).transpose(1, 2))                           # N x T x H -> N x H x T

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
        return output


class Encoder2(nn.Module):
    """ input: N x T
        spkr_vec: N x 1 x S
        return: N x T x H
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, dropout_p, spkr_embed_size, fluency=0, debug=0):
        super(Encoder2, self).__init__()
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

        self.register_parameter('token_bank', nn.Parameter(torch.randn(1, 1, int(vocab_size / 2), charvec_dim)))
        self.ref_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.token_proj = nn.Linear(charvec_dim, hidden_size, bias=False)
        self.att_proj = nn.Linear(hidden_size, 1, bias=False)

        if fluency == 0:
            self.spkr_fc = nn.Sequential(nn.Linear(spkr_embed_size, hidden_size * len(self.conv)), nn.Softsign())

    def forward(self, input, text_lengths, spkr_vec=None, debug=0):
        output = self.conv_1st(self.embedding(input).transpose(1, 2))                           # N x T x H -> N x H x T

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

        # pronunciation token
        N, T, _ = output.size()
        token_bank = self.token_bank.expand(N, T, -1, -1)  # N x T x T_tok x H_sty

        # attention -- https://arxiv.org/pdf/1506.07503.pdf
        e = self.token_proj(token_bank) + self.ref_proj(output.view(N, -1)).view(N, T, 1, -1)  # N x T x T_tok x H_att

        # stable softmax
        logit = self.att_proj(torch.tanh(e))
        logit_max, _ = torch.max(logit, dim=2, keepdim=True)
        att_weights = torch.exp(logit - logit_max)
        att_weights = F.normalize(att_weights, 1, 2)  # N x T x T_tok x 1

        att_weights = att_weights.view(N * T, 1, -1)
        token_bank = token_bank.view(N * T, token_bank.size(2), -1)
        output = torch.bmm(att_weights, torch.tanh(token_bank))  # N x 1 x H_sty
        return output


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

    def forward(self, x, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

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


class ReferenceEncoderOld(nn.Module):
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

        if debug == 12 or debug == 7:
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

    def forward(self, x, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            if debug == 12 or debug == 7:
                output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

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

            if debug == 10:
                att_weights = torch.sigmoid(att_weights)              # N x T_tok x 1
            else:
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
        return output, context

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


class AttnDecoderRNN2(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, att_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, r_factor=2, dropout_p=0.5, debug=0):
        super(AttnDecoderRNN2, self).__init__()
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

        self.LSTM = nn.LSTM(enc_hidden + dec_hidden + spkr_embed_size, 4 * dec_hidden,
                            num_layers=self.num_lstm_layers, batch_first=True)

        if debug == 13:
            from tacotron.module import WaveGlow
            self.out_linear = WaveGlow(enc_hidden + 4 * dec_hidden, output_size * r_factor, 1, 1, 2, 4, 2, 1, dec_hidden, 3)
        elif debug == 9:
            self.out_linear = nn.Linear(enc_hidden + 4 * dec_hidden + spkr_embed_size, output_size * r_factor)
        else:
            self.out_linear = nn.Linear(enc_hidden + 4 * dec_hidden, output_size * r_factor)

        self.set_attention_range(att_range)
        self.reset_states()

    def forward(self, input_enc, input_dec, spkr_vec, lengths_enc, speed, debug=0, context_vec=None, output_dec=None):
        N, T_enc = input_enc.size(0), max(lengths_enc)
        in_att_speed = self.in_att_speed(speed.unsqueeze(-1)).unsqueeze(1)

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

        e = self.att_bias_enc + in_att_dec + self.att_bias_spkr + in_att_prev_att + in_att_speed     # N x T_enc x H_att

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

        input_enc = input_enc + self.speed_proj(speed.unsqueeze(-1)).unsqueeze(1)

        context = torch.bmm(self.att_weights.transpose(1, 2), input_enc)  # N x 1 x 2H

        in_lstm = torch.cat((out_prenet, context, spkr_vec), 2)        # N x 1 x 4H

        if self.hidden is None:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, None)           # N x 1 x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, (self.hidden, self.cell))  # N x 1 x 4H, L x N x 4H, L x N x 4H

        if debug == 9:
            dec_output = torch.cat((out_lstm, context, spkr_vec), 2)                      # N x 1 x 6H
        else:
            dec_output = torch.cat((out_lstm, context), 2)                              # N x 1 x 6H

        if debug == 13:
            if self.training:
                output, log_s_list, log_W_list = self.out_linear([dec_output.squeeze(1).unsqueeze(2), output_dec])
            else:
                output = self.out_linear.infer(dec_output.squeeze(1).unsqeeze(2))
        else:
            output = self.out_linear(dec_output).view(N, self.r_factor, -1)                 # N x r x O_dec
        return output, context

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


class AttnDecoderRNN3(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_dec: Output from previous-step decoder (NxO_dec)
        spkr_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, enc_hidden, att_hidden, dec_hidden, output_size, spkr_embed_size, att_range=10, r_factor=2, dropout_p=0.5, debug=0, num_t_layer=2):
        super(AttnDecoderRNN3, self).__init__()
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

        self.LSTM = nn.LSTM(enc_hidden + dec_hidden + spkr_embed_size, 4 * dec_hidden,
                            num_layers=self.num_lstm_layers, batch_first=True)

        # self.out_linear = nn.Linear(enc_hidden + 4 * dec_hidden, output_size * r_factor)
        if debug == 12:
            self.out_linear = nn.Sequential(
                nn.TransformerEncoder(nn.TransformerEncoderLayer(enc_hidden + 4 * dec_hidden, 8, 2 * dec_hidden), 1),
                nn.Linear(enc_hidden + 4 * dec_hidden, dec_hidden),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(dec_hidden, 8, 2 * dec_hidden), num_t_layer),
                nn.Linear(dec_hidden, output_size * r_factor),
            )
        elif debug == 14 or debug == 16:
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

        e = self.att_bias_enc + in_att_dec + self.att_bias_spkr + in_att_prev_att + in_att_speed     # N x T_enc x H_att

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

        input_enc = input_enc + self.speed_proj(speed.unsqueeze(-1)).unsqueeze(1)

        context = torch.bmm(self.att_weights.transpose(1, 2), input_enc)  # N x 1 x 2H

        in_lstm = torch.cat((out_prenet, context, spkr_vec), 2)        # N x 1 x 4H

        if self.hidden is None:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, None)           # N x 1 x 4H, L x N x 4H, L x N x 4H
        else:
            out_lstm, (self.hidden, self.cell) = self.LSTM(in_lstm, (self.hidden, self.cell))  # N x 1 x 4H, L x N x 4H, L x N x 4H

        if debug == 12:
            dec_output = torch.cat((out_lstm, context), 2).transpose(0, 1)                              # N x 1 x 6H
        elif debug == 14 or debug == 16:
            dec_output = torch.cat((out_lstm, in_lstm), 2).transpose(0, 1)                              # N x 1 x 6H

        output = self.out_linear(dec_output).transpose(0, 1).view(N, self.r_factor, -1)                 # N x r x O_dec
        return output, context

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


class NewPostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, dropout_p):
        super(NewPostProcessor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dec_out_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.InstanceNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_size, post_out_size, 5, padding=2),
        )

    def forward(self, input, lengths=None):
        return self.net(input.transpose(1,2)).transpose(1,2)


class PostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, num_filters):
        super(PostProcessor, self).__init__()
        self.CBHG = CBHG(dec_out_size, hidden_size, 2 * hidden_size, hidden_size, hidden_size, num_filters, True)
        self.projection = nn.Linear(2 * hidden_size, post_out_size)

    def forward(self, input, lengths=None):
        if lengths is None:
            N, T = input.size(0), input.size(1)
            lengths = [T for _ in range(N)]
            output = self.CBHG(input, lengths).contiguous()
            output = self.projection(output)
        else:
            output = self.CBHG(input, lengths)
            output = rnn.pack_padded_sequence(output, lengths, True, enforce_sorted=False)
            output = rnn.PackedSequence(self.projection(output.data), output.batch_sizes)
            output, _ = rnn.pad_packed_sequence(output, True)
        return output


class CBHG(nn.Module):
    """ input: NxTxinput_dim sized Tensor
        output: NxTx2gru_dim sized Tensor
    """
    def __init__(self, input_dim, conv_bank_dim, conv_dim1, conv_dim2, gru_dim, num_filters, is_masked):
        super(CBHG, self).__init__()
        self.num_filters = num_filters

        bank_out_dim = num_filters * conv_bank_dim
        self.conv_bank = nn.ModuleList()
        for i in range(num_filters):
            self.conv_bank.append(nn.Conv1d(input_dim, conv_bank_dim, i + 1, stride=1, padding=int(np.ceil(i / 2))))

        # define batch normalization layer, we use BN1D since the sequence length is not fixed
        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.InstanceNorm1d(bank_out_dim))
        self.bn_list.append(nn.InstanceNorm1d(conv_dim1))
        self.bn_list.append(nn.InstanceNorm1d(conv_dim2))

        self.conv1 = nn.Conv1d(bank_out_dim, conv_dim1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv_dim1, conv_dim2, 3, stride=1, padding=1)

        if input_dim != conv_dim2:
            self.residual_proj = nn.Linear(input_dim, conv_dim2)

        self.highway = Highway(conv_dim2, 4)
        self.rnn_residual = nn.Linear(conv_dim2, 2*conv_dim2)
        self.BGRU = nn.GRU(input_size=conv_dim2, hidden_size=gru_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input, lengths, spkr_vec_list=None):
        N, T = input.size(0), input.size(1)

        if spkr_vec_list is None:
            spkr_b1, spkr_b2, spkr_b3 = 0, 0, 0
        else:
            spkr_b1, spkr_b2, spkr_b3 = spkr_vec_list

        conv_bank_out = []
        for i in range(self.num_filters):
            tmp_input = input.transpose(1, 2)  # NxTxH -> NxHxT
            if i % 2 == 0:
                tmp_input = tmp_input.unsqueeze(-1)
                tmp_input = F.pad(tmp_input, (0,0,0,1)).squeeze(-1)   # NxHxT
            conv_bank_out.append(self.conv_bank[i](tmp_input) + spkr_b1)

        residual = torch.cat(conv_bank_out, dim=1)                  # NxHFxT
        residual = F.relu(self.bn_list[0](residual))
        residual = F.max_pool1d(residual, 2, stride=1)
        residual = self.conv1(residual)                             # NxHxT
        residual = F.relu(self.bn_list[1](residual)) + spkr_b2
        residual = self.conv2(residual)                             # NxHxT
        residual = self.bn_list[2](residual) + spkr_b3
        residual = residual.transpose(1,2)                          # NxHxT -> NxTxH

        rnn_input = input
        if rnn_input.size() != residual.size():
            rnn_input = self.residual_proj(rnn_input)
        rnn_input = rnn_input + residual
        rnn_input = self.highway(rnn_input)

        output = rnn.pack_padded_sequence(rnn_input, lengths, True, enforce_sorted=False)
        output, _ = self.BGRU(output)                               # zero h_0 is used by default
        output, _ = rnn.pad_packed_sequence(output, True)           # NxTx2H

        rnn_residual = self.rnn_residual(rnn_input)
        output = rnn_residual + output
        return output


class Highway(nn.Module):
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """ input: NxH sized Tensor
            output: NxH sized Tensor
        """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


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

    def forward(self, x, spkr_vec=None, debug=0):
        """ x: N x T x O_dec sized Tensor (Spectrogram)
            output: N x (T/r_factor) x H sized Tensor
        """
        N, T_ori, O_dec = x.size(0), x.size(1), x.size(2)
        output_ref_enc = x.transpose(1, 2).unsqueeze(1)      # N x 1 x C x T

        for i in range(len(self.conv_ref_enc)):
            output_ref_enc = self.pad_SAME(output_ref_enc, self.filter_size_ref[i], self.stride_size_ref[i])
            output_ref_enc = self.conv_ref_enc[i](output_ref_enc)                           # N x H2 x C x T
            output_ref_enc = output_ref_enc + self.spkr_biases[i](spkr_vec.squeeze(1)).view(N, -1, 1, 1)

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


class ProsodyStatsGST(nn.Module):
    """
    Store some statistics of each speaker (ex. prosody, speed)
    Currently available statistics:
        - mean: mean prosody of normal sentence
        - question: mean prosody of interrogative sentence
        - speed: mean of voice speed
    """
    def __init__(self, num_spkr, prosody_dim, debug=0):
        super(ProsodyStatsGST, self).__init__()
        self.num_spkr = num_spkr
        self.stat_types = ['means', 'question',]
        if debug == 1 or debug == 13 or debug == 12 or debug == 14 or debug == 7 or debug == 15 or debug == 16 or debug == 9:
            self.stat_types.append('speed')
            self.register_buffer('speed', torch.zeros(num_spkr, 1))

        self.register_buffer('means', torch.zeros(num_spkr, prosody_dim))
        self.register_buffer('question', torch.zeros(num_spkr, prosody_dim))

        self.acc_count = {}
        self.acc_mean = {}
        self.reset_acc_states()

    def put_stats(self, spkr_id, prosody_vec, speed=None, phoneme_input=None):
        for i, s in enumerate(spkr_id):
            if not phoneme_input is None and 134 in phoneme_input[i].tolist():  # 134 is index of "?"
                self.acc_mean['question'][s] += prosody_vec[i]
                self.acc_count['question'][s] += 1
            self.acc_mean['means'][s] += prosody_vec[i]
            self.acc_count['means'][s] += 1

            if 'speed' in self.stat_types:
                self.acc_mean['speed'][s] += speed[i]
                self.acc_count['speed'][s] += 1

    def summarize_acc_states(self):
        for s in range(self.num_spkr):
            if self.acc_count['means'][s] > 0:
                self.means[s].copy_(self.acc_mean['means'][s] / self.acc_count['means'][s])
                if self.acc_count['question'][s] < 10:
                    self.question[s] = self.means[s].clone()
                else:
                    self.question[s].copy_(self.acc_mean['question'][s] / self.acc_count['question'][s])

            if 'speed' in self.stat_types and self.acc_count['speed'][s] > 0:
                self.speed[s].copy_(self.acc_mean['speed'][s] / self.acc_count['speed'][s])

        self.reset_acc_states()

    def normalize_prosody(self, prosody, spkr_id):
        center = torch.index_select(self.means, 0, spkr_id).type_as(prosody) + \
                 torch.index_select(self.question, 0, spkr_id).type_as(prosody)       # N x style_dim
        center = center.unsqueeze(1) / 2
        return prosody - center

    def reset_acc_states(self):
        for stat_type in self.stat_types:
            self.acc_count[stat_type] = [0 for _ in range(self.num_spkr)]
            self.acc_mean[stat_type] = [0 for _ in range(self.num_spkr)]


class GradReverse(Function):
    def __init__(self, lambd, max_norm, norm_type):
        self.lambd = lambd
        self.max_norm = max_norm
        self.norm_type = norm_type

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        if self.max_norm is not None:
            max_norm = float(self.max_norm)
            norm_type = float(self.norm_type)
            param_norm = grad_output.data.norm(norm_type).item()
            clip_coeff = max_norm / (param_norm + 1e-6)

            if clip_coeff < 1:
                grad_output = grad_output * clip_coeff

        return (grad_output * -self.lambd)


def grad_reverse(x, lambd, max_norm=None, norm_type=2):
    return GradReverse(lambd, max_norm, norm_type)(x)