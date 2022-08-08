# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import random

from tacotron.module.deprecated_model.model_arattention.module.Encoder import Encoder, EncoderConv
from tacotron.module.deprecated_model.model_arattention.module.ReferenceEncoder import ReferenceEncoder
from tacotron.module.deprecated_model.model_arattention.module.ProsodyStats import ProsodyStatsGST
from tacotron.module.deprecated_model.model_arattention.module.Decoder import AttnDecoderRNN2
from tacotron.module.deprecated_model.model_arattention.module.PostProcessor import PostProcessor
from tacotron.module.deprecated_model.model_arattention.module import guided_att_loss


class Tacotron(nn.Module):
    def __init__(self, args, **kwargs):
        super(Tacotron, self).__init__()
        self.trunc_size = args.trunc_size
        self.r_factor = args.r_factor
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.fluency = args.fluency
        self.dec_out_type = args.dec_out_type
        self.manual_seqend_offset = 0
        self.is_reset = True
        self.att_weights = None
        self.prev_dec_output = None
        self.detach = False
        self.variables_to_keep = set([])
        vocab = kwargs.get('vocab', {})

        if args.conv == 0:
            self.encoder = Encoder(args.vocab_size, args.charvec_dim, args.enc_hidden, args.dropout, args.spkr_embed_size, fluency=args.fluency, debug=args.debug)
        else:
            self.encoder = EncoderConv(args.vocab_size, args.charvec_dim, args.enc_hidden, args.spkr_embed_size, debug=args.debug)

        self.spkr_embed = nn.Embedding(args.num_id, args.spkr_embed_size, max_norm=2)  # randomly chosen norm size

        self.ref_encoder = ReferenceEncoder(args.dec_out_size, args.enc_hidden, args.att_hidden, args.n_token, debug=args.debug, n_head=args.n_head, spkr_embed_size=args.spkr_embed_size)
        self.prosody_stats = ProsodyStatsGST(args.num_id, args.enc_hidden, vocab=vocab, debug=args.debug)

        self.decoder = AttnDecoderRNN2(args.enc_hidden, args.att_hidden, args.dec_hidden, args.dec_out_size, args.spkr_embed_size, args.att_range, args.r_factor, args.dropout, debug=args.debug)

        self.post_processor = PostProcessor(args.att_hidden, args.dec_out_size, args.post_out_size, 8)

        # attention sharpness to measure generation quality
        self.attention_sharpness = 0

        # guided attention loss
        self.guided_att_loss = guided_att_loss.GuidedAttentionLoss()

    def forward(self, enc_input, dec_input, spkr_id, spec_lengths, text_lengths, whole_spec_len=None, spkr_vec=None, debug=0, 
                gst_vec=None, gst_source=None, gst_spkr=None, stop_type=0, speed_x=1.0, **kwargs):
        N, r = enc_input.size(0), self.r_factor
        T_wav, T_dec = min(spec_lengths), min(spec_lengths)//r
        T_enc = enc_input.size(1)
        self.att_weights = []
        taco_out_dict = {}
        spkr_adv_loss = 0

        if spkr_vec is None:
            spkr_vec = self.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        else:
            spkr_vec = spkr_vec

        # set speed
        if kwargs.get('speed') is not None:
            speed = spkr_vec.new().resize_(N).fill_(kwargs.get('speed'))
        else:
            if gst_source == 'ref_wav' or self.training:
                speed = torch.Tensor(text_lengths).type_as(spkr_vec) / whole_spec_len.float()  # N
            else:
                speed = torch.index_select(self.prosody_stats.speed, 0, spkr_id).squeeze(1)
            speed /= speed_x

        phoneme_emb, enc_output = self.encoder(enc_input, text_lengths, spkr_vec=None, debug=debug)

        # get style vector
        if gst_spkr is None:
            gst_spkr_id = spkr_id
            gst_spkr_vec = spkr_vec
        else:
            gst_spkr_id = torch.LongTensor([gst_spkr for _ in range(N)]).type_as(enc_input)
            gst_spkr_vec = self.spkr_embed(gst_spkr_id).unsqueeze(1)

        ref_dec_input = kwargs.get('target_mel_whole')
        if gst_source == 'ref_wav' or self.training:
            gst_vec, gst_att = self.ref_encoder(ref_dec_input, whole_spec_len=whole_spec_len, spkr_vec=gst_spkr_vec, debug=debug)  # N x 1 x style_dim
        elif gst_source == 'cluster':
            # Use curated style token
            gst_vec = gst_vec.view(N, 1, -1)
        elif gst_source == 'gst_mean':
            # assert (N == 1)
            # Use mean style token saved before. (Usually in evaluation)
            gst_vec_question = torch.index_select(self.prosody_stats.question, 0, gst_spkr_id).unsqueeze(1)   # N x 1 x style_dim
            gst_vec_mean = torch.index_select(self.prosody_stats.means, 0, gst_spkr_id).unsqueeze(1)          # N x 1 x style_dim

            mask_question = [1 if 134 in enc_input[i].tolist() else 0 for i in range(enc_input.size(0))]    # 134 is "?"
            mask_question = torch.Tensor(mask_question).type_as(gst_vec_question).view(N, 1, 1)
            gst_vec = mask_question * gst_vec_question + (1 - mask_question) * gst_vec_mean
        else:
            raise RuntimeError(f'Not supported style source: {gst_source}')

        dec_input_from_enc = enc_output + gst_vec
        dec_loop_out_dict = self.decoder_loop(T_dec, dec_input_from_enc, dec_input, spkr_vec, text_lengths,
                                              whole_spec_len, gst_vec, speed, debug,
                                              enc_input=enc_input, spkr_id=spkr_id, stop_type=stop_type)
        output_dec = dec_loop_out_dict.get('output_dec')
        seq_end = dec_loop_out_dict.get('seq_end')
        context_list = dec_loop_out_dict.get('context_list')
        output_dec_lstm_list = dec_loop_out_dict.get('output_dec_lstm_list')

        # compute attention sharpness to measure generation quality.
        # Note that the error is not backpropagated from attention sharpness.
        self.attention_sharpness = self.attention_sharpness.data.item() / (N * T_dec)

        output_post = self.post_processor(output_dec)

        if "att_context" in self.variables_to_keep:
            feat_context = torch.cat(context_list, dim=1)
        if "taco_lstm_out" in self.variables_to_keep:
            feat_dec_lstm = torch.cat(output_dec_lstm_list, dim=1)

        if self.training:
            att_loss = 0
            att_weights = torch.cat(self.att_weights, dim=-1).transpose(1, 2)       # N x T_dec x T_enc

            # guided attention loss
            ilens = torch.Tensor(text_lengths).type_as(enc_input)
            olens = whole_spec_len.float().div(self.r_factor).ceil().long()
            slice_from_end = not self.is_reset
            guided_att_loss = self.guided_att_loss(att_weights, ilens, olens, slice_from_end)
            att_loss += guided_att_loss

            taco_out_dict.update({"att_loss": att_loss})

            spkr_adv_loss = 0
            taco_out_dict.update({"spkr_adv_loss": spkr_adv_loss})

            # compute prenet_loss
            prenet_loss = 0
            taco_out_dict.update({"prenet_loss": prenet_loss})

        taco_out_dict.update({
            'output_dec': output_dec,
            'output_post': output_post,
            'gst_vec': gst_vec,
            'seq_end': seq_end,
        })
        if "att_context" in self.variables_to_keep:
            taco_out_dict["att_context"] = feat_context
        if "taco_lstm_out" in self.variables_to_keep:
            taco_out_dict["taco_lstm_out"] = feat_dec_lstm

        self.is_reset = False
        return taco_out_dict

    def decoder_loop(self, T_dec, dec_input_from_enc, dec_input, spkr_vec, text_lengths, whole_spec_len, style_vec, speed, debug, **kwargs):
        N, r = dec_input_from_enc.size(0), self.r_factor
        output_mel_list = []
        output_dec_lstm_list = []
        context_list = []
        spkr_id = kwargs.get('spkr_id', None)
        stop_type = kwargs.get('stop_type', 0)
        enc_input = kwargs.get('enc_input', 0)
        text_lengths = torch.LongTensor(text_lengths).to(dec_input_from_enc.device)

        # Stopping criteria
        sequence_end = 0
        if not self.training:
            if stop_type == 0:
                is_att_ended = spkr_vec.data.new().resize_(N).long().zero_()
                is_volume_ended = is_att_ended.data.new().resize_(N).zero_()
                max_mel_norm = spkr_vec.data.new().resize_(N).zero_()
                min_mel_norm = spkr_vec.data.new().resize_(N).fill_(1000)
                latest_max_position = is_att_ended.data.new().resize_(N).fill_(-1)
                end_condition = is_att_ended.data.new().resize_(N).zero_()
                text_end_position = torch.clamp(text_lengths - 1, min=0)
                expected_seq_end = torch.clamp(text_end_position, min=4)
                sequence_end = is_att_ended.data.new().resize_(N).zero_()
                att_offset = is_att_ended.data.new().resize_(N).zero_()

                # calibrate by speech speed
                speed_calib = torch.clamp(torch.ceil(1 / speed / self.r_factor) - 1, min=0)
            elif stop_type == -1:
                is_att_ended = spkr_vec.data.new().resize_(N).long().zero_()
                is_volume_ended = is_att_ended.data.new().resize_(N).zero_()
                max_mel_norm = spkr_vec.data.new().resize_(N).zero_()
                min_mel_norm = spkr_vec.data.new().resize_(N).fill_(1000)
                end_condition = is_att_ended.data.new().resize_(N).zero_()
                text_end_position = torch.clamp(text_lengths - 2, min=0)
                sequence_end = torch.Tensor([0 for _ in range(N)]).type_as(enc_input.data)
            elif stop_type == 1:
                end_buffer = 5
                text_end_position = text_lengths
                sequence_end = torch.Tensor([end_buffer for _ in range(N)]).type_as(spkr_vec.data).long()
            elif stop_type == 2:
                is_att_ended = spkr_vec.data.new().resize_(N).long().zero_()
                is_volume_ended = is_att_ended.data.new().resize_(N).zero_()
                max_mel_norm = spkr_vec.data.new().resize_(N).zero_()
                min_mel_norm = spkr_vec.data.new().resize_(N).fill_(1000)
                expected_seq_end = is_att_ended.data.new().resize_(N).fill_(4)
                latest_max_position = is_att_ended.data.new().resize_(N).fill_(1.0/4.0)      # some speakers may have silence for 3 time-steps
                end_condition = is_att_ended.data.new().resize_(N).zero_()
                text_end_position = torch.clamp(text_lengths - 1, min=0)
                sequence_end = is_att_ended.data.new().resize_(N).zero_()
                att_offset = is_att_ended.data.new().resize_(N).zero_()

                # calibrate by speech speed
                speed_calib = torch.clamp(torch.ceil(1 / speed / self.r_factor) - 1, min=0)
                end_buffer = 4

        # Generation length FIX

        self.decoder.reset_bias()
        for di in range(T_dec):
            end_idx = (di+1)*r - 1
            dec_out_dict = self.decoder(dec_input_from_enc, self.prev_dec_output, spkr_vec, text_lengths, speed,
                                        debug=debug, context_vec=None)
            self.prev_dec_output = dec_out_dict.get('output_dec')
            output_mel_list.append(self.prev_dec_output)

            if "att_context" in self.variables_to_keep:
                context_list.append(dec_out_dict.get('context'))
            if "taco_lstm_out" in self.variables_to_keep:
                output_dec_lstm_list.append(dec_out_dict.get('output_lstm'))

            # compute attention sharpness
            curr_max_weights, curr_max_position = torch.max(self.decoder.att_weights.data.view(N, -1), dim=1)
            self.attention_sharpness = self.attention_sharpness + torch.sum(1 - curr_max_weights)
            self.att_weights.append(self.decoder.att_weights)

            if not self.training:
                # Generation length FIX
                if stop_type == 0:
                    # Denormalize & Convert back to linear
                    spec = (torch.clamp(self.prev_dec_output, 0, 1) * 100) - 100
                    spec = torch.pow(10.0, spec * 0.05)

                    # Stopping criteria
                    curr_mel_norm, _ = torch.topk(torch.norm(spec, dim=2), min(2, self.r_factor), dim=1, largest=True)
                    curr_mel_norm = curr_mel_norm[:, -1]
                    buffer = torch.clamp((max_mel_norm - min_mel_norm) * 0.03, min=0)
                    if di > int(16 / self.r_factor):  # some speakers may have silence for 3 time-steps
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
                                           + att_offset.float() \
                                           + self.manual_seqend_offset
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
                elif stop_type == 2:
                    # Denormalize & Convert back to linear
                    spec = (torch.clamp(self.prev_dec_output, 0, 1) * 100) - 100
                    spec = torch.pow(10.0, spec * 0.05)

                    # Stopping criteria
                    curr_mel_norm, _ = torch.topk(torch.norm(spec, dim=2), min(2, self.r_factor), dim=1, largest=True)
                    curr_mel_norm = curr_mel_norm[:, -1]
                    buffer = torch.clamp((max_mel_norm - min_mel_norm) * 0.03, min=0)
                    if di > int(16 / self.r_factor):  # some speakers may have silence for 3 time-steps
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
                    latest_max_position = update_mask * curr_max_position + (1 - update_mask) * latest_max_position

                    new_expected_seq_end = (text_end_position.float() / (latest_max_position + 1).float()) * di \
                                           + speed_calib \
                                           + att_offset.float()
                    expected_seq_end = update_mask * torch.clamp(torch.ceil(new_expected_seq_end).long(), min=5) \
                                       + (1 - update_mask) * expected_seq_end

                    if end_condition.prod().item() == 1:
                        if end_buffer > 0:
                            sequence_end = sequence_end + 1
                            end_buffer -= 1
                        else:
                            break
                # Generation length FIX

            if random.random() < self.teacher_forcing_ratio:
                self.prev_dec_output = dec_input[:, end_idx]                       # Teacher forcing
            else:
                self.prev_dec_output = self.prev_dec_output[:, -1]

        output_dec = torch.cat(output_mel_list, dim=1)

        dec_loop_out_dict = {
            'output_dec': output_dec,
            'seq_end': (sequence_end + 1) * r,  # add 1 bc it is used for end index.
            'context_list': context_list,
            'output_dec_lstm_list': output_dec_lstm_list,
        }
        return dec_loop_out_dict

    def reset_decoder_states(self, debug=0):
        # wrapper
        self.is_reset = True
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
        if 'e' in l:
            print('freeze encoder.')
            freeze_params(self.encoder)
        if 'p' in l:
            print('freeze post processor.')
            freeze_params(self.post_processor)

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

    def keep_features(self, key):
        if key not in ["output_dec", "output_post", "taco_lstm_out", "att_context"]:
            raise RuntimeError("Unsupported variable. Choose one of the followings: taco_lstm_out, att_context")
        self.variables_to_keep.add(key)

    def set_attention_range(self, att_range):
        self.decoder.set_attention_range(att_range)

    def set_manual_seqend_offset(self, offset_value):
        self.manual_seqend_offset = offset_value