# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tacotron.module.Encoder import Encoder
from tacotron.module.ProsodyStats import ProsodyStatsGST
from tacotron.module.Attention import Attention
from tacotron.module.Decoder import DecoderRNN
from tacotron.module.DurationPredictor import DurationPredictor, TemporalPredictorRNN
from tacotron.module.commons import generate_path
from tacotron.module.PostProcessor import PostProcessor
from tacotron.module.GradReverse import grad_reverse
from tacotron.module.experimental.layers import PosteriorEncoder


class Tacotron(nn.Module):
    """
    voice conversion using sside prosody.
    """
    def __init__(self, args, **kwargs):
        super(Tacotron, self).__init__()
        self.exp_no = args.exp_no
        self.trunc_size = args.trunc_size
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.manual_seqend_offset = 0
        self.is_reset = True
        self.att_weights = None
        self.prev_dec_output = None
        self.prev_sside_prosody_ref = None
        self.detach = False
        self.aug_teacher_forcing = True if args.aug_teacher_forcing == 1 else False
        self.variables_to_keep = set([])
        self.vocab = kwargs.get('vocab', {})
        self.vocab = kwargs.get('vocab', {})
        self.idx_to_vocab = kwargs.get('idx_to_vocab', {})
        self.sside_z_size = args.prosody_size

        common_hidden = args.att_hidden
     
        self.encoder = Encoder(
            args.vocab_size,
            args.charvec_dim,
            args.enc_hidden,
            args.dropout,
            debug=args.debug
        )
        self.att_encoder = Encoder(
            args.vocab_size,
            args.charvec_dim,
            args.enc_hidden,
            args.dropout,
            debug=args.debug
        )

        self.spkr_embed = nn.Embedding(
            args.num_id,
            args.spkr_embed_size,
            max_norm=2
        )  # randomly chosen norm size

        # posteriors
        self.fine_encoder = PosteriorEncoder(
            args.dec_out_size,
            args.prosody_size,
            common_hidden,
            5,
            1,
            6,      # originally 16
            gin_channels=args.spkr_embed_size
        )
        
        # priors
        self.sside_prosody_predictor = TemporalPredictorRNN(
            args.enc_hidden,
            args.dec_hidden,
            args.prosody_size * 2,
            args.spkr_embed_size,
            0.1,
            debug=args.debug,
            ar_input_size=args.prosody_size
        )
            
        self.prosody_stats = ProsodyStatsGST(
            args.num_id,
            args.enc_hidden,
            vocab=self.vocab,
            debug=args.debug
        )

        self.attention = Attention(
            args.enc_hidden,
            args.dec_out_size,
            args.att_hidden,
            args.num_id,
            debug=args.debug,
            vocab_size=args.vocab_size
        )
        short_tokens = set(['!', "'", '.', '-', '?', '~'])
        short_tokens = torch.tensor([1 if x in short_tokens else 0 for x in self.idx_to_vocab])
        self.register_buffer('short_token', short_tokens.view(args.vocab_size ,1))

        self.duration_predictor = DurationPredictor(
            args.enc_hidden,
            args.dec_hidden,
            3,
            0.,
            args.spkr_embed_size,
            args.enc_hidden,
            args.debug
        )

        # AR decoder
        self.decoder = DecoderRNN(
            args.enc_hidden,
            args.dec_hidden,
            args.dec_out_size,
            args.spkr_embed_size,
            pitch_size=args.prosody_size,
            pitch_embed_size=16,
            dropout_p=args.dropout,
            debug=args.debug
        )

        self.post_processor = PostProcessor(
            common_hidden,
            args.dec_out_size,
            args.post_out_size,
            8
        )


    def forward(self, enc_input, dec_input, spkr_id, spec_lengths, text_lengths, 
                whole_spec_len=None, spkr_vec=None, debug=0, 
                prosody_vec=None, prosody_source=None, prosody_spkr=None,
                speed_x=1.0, **kwargs):
        N = enc_input.size(0)
        T_dec = min(spec_lengths)
        taco_out_dict = {}
        attention_ref = None
        sside_prosody = None
        tside_prosody = None
        ref_dec_input = kwargs.get('target_mel_whole')

        # masking
        text_lengths, spec_lengths, text_mask, spec_mask, whole_spec_mask = self.get_seq_mask(N, text_lengths, spec_lengths, whole_spec_len, enc_input.device)

        # MODULE: speaker
        if spkr_vec is None:
            spkr_vec = self.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        else:
            spkr_vec = spkr_vec

        if prosody_spkr is None:
            prosody_spkr_vec = spkr_vec
        else:
            prosody_spkr_id = torch.LongTensor([prosody_spkr for _ in range(N)]).type_as(enc_input)
            prosody_spkr_vec = self.spkr_embed(prosody_spkr_id).unsqueeze(1)

        # MODULE: speed
        if kwargs.get('speed') is not None:
            speed = spkr_vec.new().resize_(N).fill_(kwargs.get('speed'))
        else:
            if prosody_source == 'ref_wav' or self.training:
                speed = text_lengths.type_as(spkr_vec) / whole_spec_len.float()  # N
            else:
                speed = torch.index_select(self.prosody_stats.speed, 0, prosody_spkr_id).squeeze(1)
            speed /= speed_x

        # MODULE: encoder
        phoneme_emb, pronunciation_vec = self.encoder(
            enc_input,
            text_lengths,
            text_mask,
            debug=debug
        )
        _, att_enc_output = self.att_encoder(
            enc_input,
            text_lengths,
            text_mask,
            debug=debug
        )

        # MODULE: duration
        duration_pred = self.duration_predictor(
            att_enc_output.transpose(1, 2).detach(), 
            text_mask.unsqueeze(1),
            spkr_vec=spkr_vec.detach(),
            gst=None,
            speed=speed.detach(),
            text_lengths=text_lengths,
            debug=debug
        )

        # MODULE: alignment
        if prosody_source == 'ref_wav' or self.training:
            short_token_mask = F.embedding(enc_input, self.short_token)
            attention_ref_whole, att_loss, _, _ = self.attention(
                att_enc_output,
                ref_dec_input,
                text_lengths,
                whole_spec_len,
                debug,
                enc_input=enc_input,
                short_token_mask=short_token_mask,
            )
        else:
            w = duration_pred * text_mask * speed_x
            w_ceil = torch.round(torch.clamp(w, min=1))
            T_dec = int(w_ceil.sum().item())
            attention_mask = text_mask.unsqueeze(-1).expand(-1, -1, T_dec)
            attention_ref_whole = generate_path(w_ceil.squeeze(1), attention_mask).float()
        att_cumsum_whole = torch.cumsum(attention_ref_whole, dim=2)
        duration_gt = torch.sum(attention_ref_whole, -1)
        taco_out_dict.update({"duration": duration_gt})

        # MODULE: sside posterior
        if prosody_source == 'ref_wav' or self.training:
            sside_zq_whole, sside_mq_whole, sside_logsq_whole = self.fine_encoder(
                ref_dec_input.transpose(1, 2),
                whole_spec_mask.view(N, 1, -1),
                g=spkr_vec.view(N, -1, 1)
            )
        else:
            sside_zq_whole = sside_mq_whole = sside_logsq_whole = None

        # MODULE: sside prior
        expanded_pronunciation_vec = torch.bmm(attention_ref_whole.transpose(1, 2), pronunciation_vec)
        sside_z_padding = torch.zeros(N, 1, self.sside_z_size, device=pronunciation_vec.device, dtype=torch.float)
        if self.training:
            sside_zq_shift = torch.cat(
                [sside_z_padding, sside_zq_whole[:, :-1]], dim=1
            )

            self.sside_prosody_predictor.reset_states(debug=debug)
            sside_prosody_pred = self.sside_prosody_predictor(
                expanded_pronunciation_vec.detach(),
                sside_zq_shift.detach(),
                prosody_spkr_vec.detach(),
                debug,
            )
            sside_mp_whole, sside_logsp_whole = torch.chunk(sside_prosody_pred, 2, 2)
            sside_zp_whole = sside_mp_whole + torch.randn_like(sside_mp_whole) * torch.exp(sside_logsp_whole)
        else:
            sside_zp = []
            curr_sside_zp = sside_z_padding
            self.sside_prosody_predictor.reset_states(debug=debug)
            for i in range(expanded_pronunciation_vec.size(1)):
                curr_pred = self.sside_prosody_predictor(
                    expanded_pronunciation_vec[:, i:i+1],
                    curr_sside_zp,
                    prosody_spkr_vec,
                    debug,
                )
                curr_sside_mp, curr_sside_logsp = torch.chunk(curr_pred, 2, 2)
                curr_sside_zp = curr_sside_mp + torch.randn_like(curr_sside_mp) * torch.exp(curr_sside_logsp)
                sside_zp.append(curr_sside_zp)
            sside_zp_whole = torch.cat(sside_zp, dim=1)

        if self.training:
            sside_prosody = sside_zq_whole
        elif prosody_source == 'prosody_vec':
            sside_prosody = prosody_vec
        elif prosody_source == 'prediction':
            sside_prosody = sside_zp_whole
        elif prosody_source == 'ref_wav':
            sside_prosody = sside_zq_whole
        else:
            assert False

        # MODULE: decoder
        T_dec = attention_ref_whole.size(2)
        if self.training:
            # decoder input preparation
            dec_input_padding = dec_input.data.new().resize_(N, 1, dec_input.size(2)).zero_()
            dec_input_shift_whole = torch.cat([dec_input_padding, ref_dec_input[:, :-1]], dim=1)
            if self.is_reset:
                dec_input_shift = dec_input_shift_whole[:, :self.trunc_size]
                attention_ref = attention_ref_whole[:, :, :self.trunc_size]
                att_cumsum = att_cumsum_whole[:, :, :self.trunc_size]
                sside_zq = sside_zq_whole[:, :self.trunc_size]
            else:
                dec_input_shift = dec_input_shift_whole[:, self.trunc_size:]
                attention_ref = attention_ref_whole[:, :, self.trunc_size:]
                att_cumsum = att_cumsum_whole[:, :, self.trunc_size:]
                sside_zq = sside_zq_whole[:, self.trunc_size:]

            # positonal encoding for attention
            with torch.no_grad():
                attention_ref = attention_ref[:, :, :dec_input_shift.size(1)]
                att_cumsum = att_cumsum[:, :, :dec_input_shift.size(1)]
                
                att_position = att_cumsum / torch.clamp(att_cumsum_whole[:, :, -1:], min=1) * attention_ref
                att_position = torch.sum(att_position, dim=1).unsqueeze(2)
        
            dec_out_dict = self.decoder(
                pronunciation_vec,
                dec_input_shift,
                spkr_vec,
                debug,
                attention_ref=attention_ref,
                att_position=att_position,
                sside_prosody=sside_zq
            )
            output_dec = dec_out_dict.get('output_dec')

            # teacher-forcing with tacotron output
            if self.aug_teacher_forcing:
                if self.is_reset:
                    dec_pred_shift = torch.cat([dec_input_padding, output_dec[:, :-1]], dim=1)
                else:
                    dec_pred_shift = torch.cat([dec_input_shift[:, 0:1], output_dec[:, :-1]], dim=1)
                dec_out_dict2 = self.decoder(
                    pronunciation_vec, 
                    dec_pred_shift.detach(), 
                    spkr_vec,
                    debug,
                    attention_ref=attention_ref,
                    att_position=att_position,
                    sside_prosody=sside_zq,
                    is_aug=True
                )
                output_dec2 = dec_out_dict2.get('output_dec')

            # prenet loss
            output_prenet1 = dec_out_dict.get('output_prenet')
            if self.aug_teacher_forcing:
                output_prenet2 = dec_out_dict2.get('output_prenet')
            else:
                if self.is_reset:
                    dec_pred_shift = torch.cat([dec_input_padding, output_dec[:, :-1]], dim=1)
                else:
                    dec_pred_shift = torch.cat([dec_input_shift[:, 0:1], output_dec[:, :-1]], dim=1)
                output_prenet2 = self.decoder.prenet(dec_pred_shift.detach())
            prenet_loss = torch.nn.functional.mse_loss(output_prenet1, output_prenet2)
        else:
            attention_ref = attention_ref_whole
            att_cumsum = att_cumsum_whole
            att_position = att_cumsum / att_cumsum[:, :, -1:] * attention_ref
            att_position = torch.sum(att_position, dim=1).unsqueeze(2)

            output_dec_list = []
            for i in range(T_dec):
                if i == 0:
                    curr_dec_input = torch.zeros(N, 1, 120, device=spkr_vec.device)
                else:
                    curr_dec_input = prev_dec_output[:, -1:]

                curr_att_weight = attention_ref[:, :, i:i+1]
                curr_att_position = att_position[:, i:i+1]
                dec_out_dict = self.decoder(
                    pronunciation_vec,
                    curr_dec_input,
                    spkr_vec,
                    debug,
                    attention_ref=curr_att_weight,
                    att_position=curr_att_position,
                    sside_prosody=sside_prosody[:, i:i+1],
                )
                prev_dec_output = dec_out_dict.get('output_dec')
                output_dec_list.append(prev_dec_output)
            output_dec = torch.cat(output_dec_list, dim=1)

        # MODULE: postprocessor
        post_mask = spec_mask.unsqueeze(1)
        output_post = self.post_processor(output_dec, spec_mask=post_mask) + output_dec
        if self.training and self.aug_teacher_forcing:
            output_post2 = self.post_processor(output_dec2, spec_mask=post_mask) + output_dec2
            taco_out_dict.update({
                "output_dec2": output_dec2,
                "output_post2": output_post2
            })

        # MODULE: compute additional loss
        if self.training:
            prosody_ignore_mask = kwargs.get('prosody_ignore_mask')
            if prosody_ignore_mask is not None:
                prosody_ignore_mask = torch.index_select(prosody_ignore_mask, 0, spkr_id).type_as(output_post)
                prosody_ignore_mask1 = prosody_ignore_mask.view(N, 1)
                prosody_ignore_mask2 = prosody_ignore_mask.view(N, 1, 1)
            else:
                prosody_ignore_mask = 1
                prosody_ignore_mask1 = 1
                prosody_ignore_mask2 = 1

            # spkr adv loss
            spkr_adv_loss = 0
            taco_out_dict.update({"spkr_adv_loss": spkr_adv_loss})

            # prenet_loss
            taco_out_dict.update({"prenet_loss": prenet_loss})

            # aligner_loss
            taco_out_dict.update({"att_loss": att_loss})

            # duration_loss
            denominator = torch.sum(text_lengths * prosody_ignore_mask)
            duration_loss = torch.sum((duration_pred - duration_gt)**2 * prosody_ignore_mask1) / torch.sum(text_lengths) * 0.01
            taco_out_dict.update({"durpred_loss": duration_loss})

            # pitch prediction loss
            sside_prosody_loss = 0
            denominator = torch.sum(whole_spec_len.view(N, 1, 1) * prosody_ignore_mask2 * self.sside_z_size)
            sside_prosody_loss0 = sside_logsp_whole - sside_logsq_whole - 0.5 + 0.5 * (
                (torch.exp(2 * sside_logsq_whole) + (sside_mq_whole - sside_mp_whole).pow(2)) * torch.exp(-2 * sside_logsp_whole)
            )
            sside_prosody_loss = torch.sum(sside_prosody_loss0 * whole_spec_mask.unsqueeze(2) * prosody_ignore_mask2) / denominator
            sside_prosody_loss *= 1e-2

            # if debug == xx:
            #     target_pitch = kwargs.get('target_pitch_whole').unsqueeze(2)    # N x T_dec x 1
            #     denominator = torch.sum(whole_spec_len.view(N, 1, 1) * prosody_ignore_mask2)
            #     sside_prosody_loss1 = sside_logsp_whole[:, :, 0:1] + 0.5 * (
            #         (target_pitch - sside_mp_whole[:, :, 0:1]).pow(2) * torch.exp(-2 * sside_logsp_whole[:, :, 0:1])
            #     )
            #     sside_prosody_loss1 = torch.sum(sside_prosody_loss1 * whole_spec_mask.unsqueeze(2) * prosody_ignore_mask2) / denominator
            #     sside_prosody_loss2 = sside_logsq_whole[:, :, 0:1] + 0.5 * (\
            #         (target_pitch - sside_mq_whole[:, :, 0:1]).pow(2) * torch.exp(-2 * sside_logsq_whole[:, :, 0:1])
            #     )
            #     sside_prosody_loss2 = torch.sum(sside_prosody_loss2 * whole_spec_mask.unsqueeze(2) * prosody_ignore_mask2) / denominator
            #     sside_prosody_loss += sside_prosody_loss1 + sside_prosody_loss2
            taco_out_dict.update({"sside_prosody_loss": sside_prosody_loss})

        # update output dictionary
        self.att_weights = attention_ref_whole                    # N x T_enc x T_dec
        taco_out_dict.update({
            'output_dec': output_dec,
            'output_post': output_post,
            's_prosody_vec': sside_prosody,
            't_prosody_vec': tside_prosody,
            'seq_end': spec_lengths,
        })
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
            freeze_params(self.encoder.embedding)
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
        if key not in ["output_dec", "taco_lstm_out", "att_context"]:
            raise RuntimeError("Unsupported variable. Choose one of the followings: taco_lstm_out, att_context")
        self.variables_to_keep.add(key)

    def get_seq_mask(self, batch_size, text_lengths, spec_lengths, whole_spec_lengths=None, device=None):            
        with torch.no_grad():
            text_lengths = torch.tensor(text_lengths, device=device)
            spec_lengths = torch.tensor(spec_lengths, device=device)
            if whole_spec_lengths is None:
                whole_spec_lengths = spec_lengths

            T_enc = max(text_lengths)
            T_dec = max(spec_lengths)
            T_dec_whole = max(whole_spec_lengths)
            text_mask = torch.arange(0, T_enc, device=device).view(1, -1).expand(batch_size, -1)
            text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, T_enc))             # N x T_enc
            spec_mask = torch.arange(0, T_dec, device=device).view(1, -1).expand(batch_size, -1)
            spec_mask = torch.lt(spec_mask, spec_lengths.view(-1, 1).expand(-1, T_dec))             # N x T_dec
            whole_spec_mask = torch.arange(0, T_dec_whole, device=device).view(1, -1).expand(batch_size, -1)
            whole_spec_mask = torch.lt(whole_spec_mask, spec_lengths.view(-1, 1).expand(-1, T_dec_whole))             # N x T_dec
        return text_lengths, spec_lengths, text_mask, spec_mask, whole_spec_mask

    def align(self, enc_input, text_lengths, dec_input, spec_lengths, debug):
        # TODO: this function should replace the same procedure in forward().
        N = enc_input.size(0)
        # masking
        text_lengths, spec_lengths, text_mask, _, _ = self.get_seq_mask(N, text_lengths, spec_lengths, None, enc_input.device)

        # MODULE: encoder
        _, enc_output = self.att_encoder(
            enc_input,
            text_lengths,
            text_mask,
            debug=debug
        )

        # MODULE: alignment
        short_token_mask = F.embedding(enc_input, self.short_token)
        attention, att_loss, _, nll = self.attention(
            enc_output,
            dec_input,
            text_lengths,
            spec_lengths,
            debug,
            enc_input=enc_input,
            short_token_mask=short_token_mask,
        )
        return {
            'attention': attention,
            'att_nll': nll,
            'att_loss': att_loss,
            'att_key': None,
        }