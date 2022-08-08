# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import random

from tacotron.module.deprecated_model.model_align_v1.module.Encoder import Encoder
from tacotron.module.deprecated_model.model_align_v1.module.ReferenceEncoder import ReferenceEncoder
from tacotron.module.deprecated_model.model_align_v1.module.ProsodyStats import ProsodyStatsGST
from tacotron.module.deprecated_model.model_align_v1.module.Decoder import AttnDecoderRNN2, DecoderRNN
from tacotron.module.deprecated_model.model_align_v1.module.experimental.Attention import Attention, DurationPredictor
from tacotron.module.deprecated_model.model_align_v1.module.experimental.commons import generate_path
from tacotron.module.deprecated_model.model_align_v1.module.PostProcessor import ConvPostProcessor, PostProcessor
from tacotron.module.deprecated_model.model_align_v1.module import guided_att_loss

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
        self.aug_teacher_forcing = True if args.aug_teacher_forcing == 1 else False
        self.train_durpred = True if args.train_durpred == 1 else False
        self.variables_to_keep = set([])
        vocab = kwargs.get('vocab', {})

        self.encoder = Encoder(args.vocab_size, args.charvec_dim, args.enc_hidden, args.dropout, args.spkr_embed_size, fluency=args.fluency, debug=args.debug)

        self.spkr_embed = nn.Embedding(args.num_id, args.spkr_embed_size, max_norm=2)  # randomly chosen norm size

        self.ref_encoder = ReferenceEncoder(args.dec_out_size, args.enc_hidden, args.att_hidden, args.n_token, debug=args.debug, n_head=args.n_head, spkr_embed_size=args.spkr_embed_size)
        self.prosody_stats = ProsodyStatsGST(args.num_id, args.enc_hidden, vocab=vocab, debug=args.debug)

        self.attention = Attention(args.enc_hidden, args.dec_out_size, args.att_hidden, args.r_factor, debug=args.debug)
        self.duration_predictor = DurationPredictor(args.enc_hidden, args.dec_hidden, 3, 0., args.spkr_embed_size, args.debug)

        # AR decoder
        self.decoder = DecoderRNN(args.enc_hidden, args.att_hidden, args.dec_hidden, args.dec_out_size, args.spkr_embed_size, args.att_range, args.r_factor, args.dropout, debug=args.debug)

        if args.conv == 1:
            self.post_processor = ConvPostProcessor(args.att_hidden, args.dec_out_size, args.post_out_size, args.dropout)
        else:
            self.post_processor = PostProcessor(args.att_hidden, args.dec_out_size, args.post_out_size, 8)

        # attention sharpness to measure generation quality
        self.attention_sharpness = 0

    def forward(self, enc_input, dec_input, spkr_id, spec_lengths, text_lengths, whole_spec_len=None, spkr_vec=None, debug=0, 
                gst_vec=None, gst_source=None, gst_spkr=None, stop_type=0, speed_x=1.0, **kwargs):
        N, r = enc_input.size(0), self.r_factor
        T_wav, T_dec = min(spec_lengths), min(spec_lengths)//r
        T_enc = enc_input.size(1)
        self.att_weights = []
        taco_out_dict = {}
        spkr_adv_loss = 0
        prenet_loss = 0

        # set speaker
        if spkr_vec is None:
            spkr_vec = self.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        else:
            spkr_vec = spkr_vec

        # set speed
        text_lengths = torch.LongTensor(text_lengths).to(enc_input)
        if kwargs.get('speed') is not None:
            speed = spkr_vec.new().resize_(N).fill_(kwargs.get('speed'))
        else:
            if gst_source == 'ref_wav' or self.training:
                speed = text_lengths.type_as(spkr_vec) / whole_spec_len.float()  # N
            else:
                speed = torch.index_select(self.prosody_stats.speed, 0, spkr_id).squeeze(1)
            speed /= speed_x

        # encode text
        phoneme_emb, enc_output = self.encoder(enc_input, text_lengths, debug=debug)

        # get style vector info
        if gst_spkr is None:
            gst_spkr_id = spkr_id
            gst_spkr_vec = spkr_vec
        else:
            gst_spkr_id = torch.LongTensor([gst_spkr for _ in range(N)]).type_as(enc_input)
            gst_spkr_vec = self.spkr_embed(gst_spkr_id).unsqueeze(1)

        ref_dec_input = kwargs.get('target_mel_whole')
        if gst_source == 'ref_wav' or self.training:
            # Extract GST
            ref_dec_output = self.ref_encoder(ref_dec_input, whole_spec_len=whole_spec_len, spkr_vec=gst_spkr_vec, debug=debug)  # N x 1 x style_dim
            gst_vec = ref_dec_output['gst']
        elif gst_source == 'cluster':
            # Use curated style token
            gst_vec = gst_vec.view(N, 1, -1)
        elif gst_source == 'gst_mean':
            # Use mean style token saved before. (Usually in evaluation)
            gst_vec_question = torch.index_select(self.prosody_stats.question, 0, gst_spkr_id).unsqueeze(1)   # N x 1 x style_dim
            gst_vec_mean = torch.index_select(self.prosody_stats.means, 0, gst_spkr_id).unsqueeze(1)          # N x 1 x style_dim

            mask_question = [1 if 134 in enc_input[i].tolist() else 0 for i in range(enc_input.size(0))]    # 134 is "?"
            mask_question = torch.Tensor(mask_question).type_as(gst_vec_question).view(N, 1, 1)
            gst_vec = mask_question * gst_vec_question + (1 - mask_question) * gst_vec_mean
        else:
            raise RuntimeError(f'Not supported style source: {gst_source}')

        # predict duration
        dec_input_from_enc = durpred_input = enc_output + gst_vec
        if self.train_durpred or not self.training:
            text_mask = torch.arange(0, enc_output.size(1), device=enc_output.device).view(1, -1).expand(N, -1)
            text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, enc_output.size(1)))                     # N x T_enc
            duration_pred = self.duration_predictor(
                    durpred_input.transpose(1, 2).detach(), 
                    text_mask.unsqueeze(1),
                    spkr_vec=spkr_vec.detach(),
                    speed=speed.detach(),
                    text_lengths=text_lengths,
                    debug=debug
                )

        if self.training:
            # attention
            attention_ref_whole, attention_loss, attention_mask, self.attention_sharpness, self.att_nll = self.attention(enc_output, ref_dec_input, text_lengths, whole_spec_len, debug)
            att_cumsum_whole = torch.cumsum(attention_ref_whole, dim=2)
            
            # decoder
            dec_input_padding = dec_input.data.new().resize_(N, 1, dec_input.size(2)).zero_()
            dec_input_shift_whole = torch.cat([dec_input_padding, ref_dec_input[:, ::r][:, :-1]], dim=1)
            if self.is_reset:
                dec_input_shift = dec_input_shift_whole[:, :self.trunc_size // r]
                attention_ref = attention_ref_whole[:, :, :self.trunc_size // r]
                att_cumsum = att_cumsum_whole[:, :, :self.trunc_size // r]
            else:
                dec_input_shift = dec_input_shift_whole[:, self.trunc_size // r:]
                attention_ref = attention_ref_whole[:, :, self.trunc_size // r:]
                att_cumsum = att_cumsum_whole[:, :, self.trunc_size // r:]

            with torch.no_grad():
                attention_ref = attention_ref[:, :, :dec_input_shift.size(1)]
                att_cumsum = att_cumsum[:, :, :dec_input_shift.size(1)]
                
                att_position = att_cumsum / torch.clamp(att_cumsum_whole[:, :, -1:], min=1) * attention_ref
                att_position = torch.sum(att_position, dim=1).unsqueeze(2)

            if debug == 95:
                spec_lengths = torch.div(whole_spec_len, self.r_factor, rounding_mode='trunc')
                pos_enc = torch.arange(0, attention_ref.size(2), device=attention_ref.device).view(1, -1).expand(N, -1)
                spec_mask = torch.lt(pos_enc, spec_lengths.view(-1, 1).expand(-1, attention_ref.size(2)))                 # N x T_dec
                pos_enc = pos_enc * spec_mask
                dec_loop_out_dict = self.decoder(dec_input_from_enc, pos_enc, spkr_vec, attention_ref)
            else:
                prev_states = (self.decoder.hidden, self.decoder.cell)
                dec_loop_out_dict = self.decoder(
                        dec_input_from_enc, 
                        dec_input_shift, 
                        spkr_vec,
                        text_lengths,
                        speed,
                        debug,
                        attention_ref=attention_ref,
                        att_position=att_position
                    )
            output_dec = dec_loop_out_dict.get('output_dec')
            seq_end = spec_lengths
            self.att_weights = attention_ref                    # N x T_enc x T_dec

            # teacher-forcing with tacotron output
            if self.aug_teacher_forcing:
                if self.is_reset:
                    dec_pred_shift = torch.cat([dec_input_padding, output_dec[:, ::r][:, :-1]], dim=1)
                else:
                    dec_pred_shift = torch.cat([dec_input_shift[:, 0:1], output_dec[:, ::r][:, :-1]], dim=1)
                dec_loop_out_dict2 = self.decoder(
                        dec_input_from_enc, 
                        dec_pred_shift.detach(), 
                        spkr_vec,
                        text_lengths,
                        speed,
                        debug,
                        attention_ref=attention_ref,
                        att_position=att_position,
                        is_aug=True
                    )
                output_dec2 = dec_loop_out_dict2.get('output_dec')

            # prenet loss
            if debug == 95:
                prenet_loss = 0
            else:
                output_prenet1 = dec_loop_out_dict.get('output_prenet')
                if self.aug_teacher_forcing:
                    output_prenet2 = dec_loop_out_dict2.get('output_prenet')
                else:
                    if self.is_reset:
                        dec_pred_shift = torch.cat([dec_input_padding, output_dec[:, ::r][:, :-1]], dim=1)
                    else:
                        dec_pred_shift = torch.cat([dec_input_shift[:, 0:1], output_dec[:, ::r][:, :-1]], dim=1)
                    output_prenet2 = self.decoder.prenet(dec_pred_shift.detach())
                prenet_loss = torch.nn.functional.mse_loss(output_prenet1, output_prenet2)

            # duration loss if attention is reliable
            if self.train_durpred and self.attention_sharpness < 0.2:
                duration_gt = torch.sum(attention_ref_whole, -1)
                duration_loss = torch.sum((duration_pred - duration_gt)**2) / torch.sum(text_lengths) * 0.01
            else:
                duration_loss = 0
        else:
            w = duration_pred * text_mask * speed_x
            w_ceil = torch.ceil(torch.clamp(w, min=1))
            T_dec = int(w_ceil.sum().item())
            y_lengths = torch.clamp_min(torch.sum(w_ceil, 1), 1).long()
            y_max_length = None
            attn_mask = text_mask.unsqueeze(-1).expand(-1, -1, T_dec)
            attention = generate_path(w_ceil.squeeze(1), attn_mask).float()

            att_cumsum = torch.cumsum(attention, dim=2)
            att_position = att_cumsum / att_cumsum[:, :, -1:] * attention
            att_position = torch.sum(att_position, dim=1).unsqueeze(2)

            if debug == 95:
                spec_len = torch.div(whole_spec_len, self.r_factor, rounding_mode='trunc')
                pos_enc = torch.arange(0, attention.size(2), device=attention.device).view(1, -1).expand(N, -1)
                spec_mask = torch.lt(pos_enc, spec_len.view(-1, 1).expand(-1, attention.size(2)))                 # N x T_dec
                pos_enc = pos_enc * spec_mask
                dec_loop_out_dict = self.decoder(dec_input_from_enc, pos_enc, spkr_vec, attention)
                output_dec = dec_loop_out_dict.get('output_dec')
            else:
                output_dec_list = []
                for i in range(T_dec):
                    if i == 0:
                        curr_dec_input = torch.zeros(N, 1, 120).to(dec_input_from_enc)
                    else:
                        curr_dec_input = prev_dec_output[:, -1:]

                    curr_att_weight = attention[:, :, i:i+1]
                    curr_att_position = att_position[:, i:i+1]
                    dec_loop_out_dict = self.decoder(
                        dec_input_from_enc, 
                        curr_dec_input, 
                        spkr_vec,
                        text_lengths,
                        speed,
                        debug,
                        attention_ref=curr_att_weight,
                        att_position=curr_att_position
                    )
                    prev_dec_output = dec_loop_out_dict.get('output_dec')
                    output_dec_list.append(prev_dec_output)
                output_dec = torch.cat(output_dec_list, dim=1)

            seq_end = torch.LongTensor(spec_lengths)
            self.att_weights = attention

        # postprocessor
        if self.dec_out_type == 'mel':
            if self.training:
                spec_mask = attention_mask[:, 0:1]
            else:
                spec_mask = None
            output_post = self.post_processor(output_dec, spec_mask=spec_mask) + output_dec
            if self.training and self.aug_teacher_forcing:
                output_post2 = self.post_processor(output_dec2, spec_mask=spec_mask) + output_dec2
                taco_out_dict.update({
                    "output_dec2": output_dec2,
                    "output_post2": output_post2
                })
        else:
            output_post = self.post_processor(output_dec, spec_mask=spec_mask)

        # compute additional loss
        if self.training:
            att_loss = 0
            att_weights = self.att_weights.transpose(1, 2)

            # attention loss
            att_loss += duration_loss + attention_loss
            taco_out_dict.update({"att_loss": att_loss})

            spkr_adv_loss = 0
            taco_out_dict.update({"spkr_adv_loss": spkr_adv_loss})

            # prenet_loss
            taco_out_dict.update({"prenet_loss": prenet_loss})

        # update output dictionary
        taco_out_dict.update({
            'output_dec': output_dec,
            'output_post': output_post,
            'gst_vec': gst_vec,
            'seq_end': seq_end,
        })
        if "att_context" in self.variables_to_keep:
            feat_context = torch.cat(context_list, dim=1)
            taco_out_dict["att_context"] = feat_context
        if "taco_lstm_out" in self.variables_to_keep:
            feat_dec_lstm = torch.cat(output_dec_lstm_list, dim=1)
            taco_out_dict["taco_lstm_out"] = feat_dec_lstm

        self.is_reset = False
        return taco_out_dict

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
        if 'a' in l:
            print('freeze attention.')
            freeze_params(self.attention)
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
        pass

    def set_manual_seqend_offset(self, offset_value):
        pass
