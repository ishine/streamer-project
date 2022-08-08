# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

from tacotron.module.deprecated_model.model_cats_v3_fluent.module.Encoder import Encoder
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.ReferenceEncoder import MelStyleEncoder
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.ProsodyStats import ProsodyStatsGST
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.Attention import Attention
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.Decoder import DecoderRNN
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.DurationPredictor import DurationPredictor
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.commons import generate_path
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.PostProcessor import PostProcessor
from tacotron.module.deprecated_model.model_cats_v3_fluent.module.GradReverse import GradientReversal

class Tacotron(nn.Module):
    def __init__(self, args, **kwargs):
        super(Tacotron, self).__init__()
        self.trunc_size = args.trunc_size
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.fluency = args.fluency
        self.manual_seqend_offset = 0
        self.is_reset = True
        self.att_weights = None
        self.prev_dec_output = None
        self.detach = False
        self.aug_teacher_forcing = True if args.aug_teacher_forcing == 1 else False
        self.variables_to_keep = set([])
        self.vocab = kwargs.get('vocab', {})
        self.idx_to_vocab = kwargs.get('idx_to_vocab', {})

        common_hidden = args.att_hidden

        lang_vec_dim = spkr_vec_dim = args.spkr_embed_size // 2

        self.encoder = Encoder(
            args.vocab_size,
            args.charvec_dim,
            args.enc_hidden,
            args.dropout,
            debug=args.debug,
            lang_vec_dim=lang_vec_dim
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
            spkr_vec_dim,
            max_norm=2
        )  # randomly chosen norm size
        self.lang_embed = nn.Embedding(
            2,
            lang_vec_dim,
            max_norm=2
        )  # randomly chosen norm size

        self.spkr_classifier = nn.Sequential(
                nn.Linear(spkr_vec_dim, spkr_vec_dim * 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(spkr_vec_dim * 2, spkr_vec_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(spkr_vec_dim, args.num_id),
            )

        self.adv_lang_classifier = nn.Sequential(
                GradientReversal(),
                nn.Linear(spkr_vec_dim, spkr_vec_dim * 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(spkr_vec_dim * 2, spkr_vec_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(spkr_vec_dim, 2),
            ) 

        self.ref_encoder = MelStyleEncoder(
            args.dec_out_size,
            args.enc_hidden,
            common_hidden,
            spkr_vec_dim,
            debug=args.debug,
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
            spkr_vec_dim,
            args.enc_hidden,
            args.debug
        )

        # AR decoder
        self.decoder = DecoderRNN(
            args.enc_hidden,
            args.dec_hidden,
            args.dec_out_size,
            spkr_vec_dim,
            pitch_size=1,
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
                gst_vec=None, gst_source=None, gst_spkr=None, 
                speed_x=1.0, **kwargs):
        N = enc_input.size(0)
        T_dec = min(spec_lengths)
        self.att_weights = []
        taco_out_dict = {}
        spkr_adv_loss = 0
        prenet_loss = 0

        # masking
        text_lengths, spec_lengths, text_mask, spec_mask, whole_spec_mask = self.get_seq_mask(N, text_lengths, spec_lengths, whole_spec_len, enc_input.device)

        # MODULE: speaker
        if spkr_vec is None:
            spkr_vec = self.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        else:
            spkr_vec = spkr_vec

        lang_id = kwargs.get('lang_id')
        # TODO: get lang_id from saved dict
        if lang_id is None:
            lang_id = torch.prod(
                        torch.logical_or(torch.lt(enc_input, 71), torch.gt(enc_input, 119)),
                        dim=1
                    ).long()
        lang_vec = self.lang_embed(lang_id).unsqueeze(1)

        if gst_spkr is None:
            gst_spkr_id = spkr_id
            gst_spkr_vec = spkr_vec
        else:
            gst_spkr_id = torch.LongTensor([gst_spkr for _ in range(N)]).type_as(enc_input)
            gst_spkr_vec = self.spkr_embed(gst_spkr_id).unsqueeze(1)

        # MODULE: speed
        if kwargs.get('speed') is not None:
            speed = spkr_vec.new().resize_(N).fill_(kwargs.get('speed'))
        else:
            if gst_source == 'ref_wav' or self.training:
                speed = text_lengths.type_as(spkr_vec) / whole_spec_len.float()  # N
            else:
                speed = torch.index_select(self.prosody_stats.speed, 0, gst_spkr_id).squeeze(1)
            speed /= speed_x

        # MODULE: reference encoder
        ref_dec_input = kwargs.get('target_mel_whole')
        if gst_source == 'ref_wav' or self.training:
            # Extract GST
            ref_enc_dict = self.ref_encoder(
                ref_dec_input,
                whole_spec_mask,
                gst_spkr_vec,
                debug=debug
            )   # N x 1 x style_dim
            gst_vec = ref_enc_dict['gst']
        elif gst_source == 'cluster':
            # Use curated style token
            gst_vec = gst_vec.view(N, 1, -1)
        elif gst_source == 'gst_mean':
            # Use mean style token saved before. (Usually in evaluation)
            id_question_mark = self.vocab['?']
            gst_vec_question = torch.index_select(self.prosody_stats.question, 0, gst_spkr_id).unsqueeze(1)   # N x 1 x style_dim
            gst_vec_mean = torch.index_select(self.prosody_stats.means, 0, gst_spkr_id).unsqueeze(1)          # N x 1 x style_dim

            mask_question = [1 if id_question_mark in enc_input[i].tolist() else 0 for i in range(enc_input.size(0))]
            mask_question = torch.Tensor(mask_question).type_as(gst_vec_question).view(N, 1, 1)
            gst_vec = mask_question * gst_vec_question + (1 - mask_question) * gst_vec_mean
        else:
            raise RuntimeError(f'Not supported style source: {gst_source}')

        # MODULE: encoder
        phoneme_emb, enc_output, _ = self.encoder(
            enc_input,
            text_lengths,
            text_mask,
            lang_vec=lang_vec,
            debug=debug
        )
        _, att_enc_output, _ = self.att_encoder(
            enc_input,
            text_lengths,
            text_mask,
            debug=debug
        )
        enc_with_gst = enc_output + gst_vec

        # MODULE: duration
        duration_pred = self.duration_predictor(
            att_enc_output.transpose(1, 2).detach(), 
            text_mask.unsqueeze(1),
            spkr_vec=spkr_vec.detach(),
            gst=gst_vec.detach(),
            speed=speed.detach(),
            text_lengths=text_lengths,
            debug=debug
        )

        # MODULE: alignment
        if self.training:
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

        # MODULE: pitch conditioner
        T_dec = attention_ref_whole.size(2)
        if self.training:
            target_pitch_whole = kwargs.get('target_pitch_whole').unsqueeze(2)          # N x T_dec x 1

            # pooling pitch to control in text-side
            pitch_flag = torch.ne(target_pitch_whole, 0).float()
            pooled_count = torch.bmm(attention_ref_whole, pitch_flag)
            pooled_pitch = torch.bmm(attention_ref_whole, target_pitch_whole) / torch.clamp(pooled_count, min=1)
            if self.training:
                prosody_ignore_mask = kwargs.get('prosody_ignore_mask')
                if prosody_ignore_mask is not None:
                    prosody_ignore_mask = torch.index_select(prosody_ignore_mask, 0, spkr_id).type_as(pooled_pitch)
                    prosody_ignore_mask = prosody_ignore_mask.view(N, 1, 1)
                else:
                    prosody_ignore_mask = 1
                pooled_pitch = pooled_pitch * torch.ge(
                    torch.rand_like(pooled_pitch) + (1-prosody_ignore_mask),
                    0.5
                )

            # expand it again (the decoder requires expanded input)
            target_pitch_whole = torch.bmm(attention_ref_whole.transpose(1, 2), pooled_pitch)
        else:
            target_pitch_whole = torch.zeros(N, T_dec, 1, device=enc_with_gst.device, dtype=torch.float)

            # force pitch when last_pitch_level is given.
            last_pitch_level = kwargs.get('last_pitch_level')     # 0.55 for question
            if last_pitch_level is not None:
                vowels = kwargs.get('vowels')
                high_pitch = self.prosody_stats.max_pitch[spkr_id].item() * last_pitch_level
                pos_last_vowel = 0
                for i in range(enc_input.size(1), 0, -1):
                    curr_idx = i-1
                    v = self.idx_to_vocab[enc_input[0, curr_idx].item()]
                    if v in vowels:
                        pos_last_vowel = max(curr_idx, pos_last_vowel)
                        break
                target_pitch_whole = torch.sum(attention_ref_whole[:, pos_last_vowel:], dim=1).unsqueeze(2) * high_pitch

            # PRINT PHONEME-PITCH PAIR
            # ph_id = torch.bmm(attention_ref_whole.transpose(1, 2), enc_input.unsqueeze(2).float()).long()
            # for i in range(target_pitch_whole.size(1)):
            #     print(i, self.idx_to_vocab[ph_id[0,i].item()], target_pitch_whole[0,i,0].item())

        # MODULE: decoder
        if self.training:
            # decoder input preparation
            dec_input_padding = dec_input.data.new().resize_(N, 1, dec_input.size(2)).zero_()
            dec_input_shift_whole = torch.cat([dec_input_padding, ref_dec_input[:, :-1]], dim=1)
            if self.is_reset:
                dec_input_shift = dec_input_shift_whole[:, :self.trunc_size]
                attention_ref = attention_ref_whole[:, :, :self.trunc_size]
                att_cumsum = att_cumsum_whole[:, :, :self.trunc_size]
                pitch = target_pitch_whole[:, :self.trunc_size]
            else:
                dec_input_shift = dec_input_shift_whole[:, self.trunc_size:]
                attention_ref = attention_ref_whole[:, :, self.trunc_size:]
                att_cumsum = att_cumsum_whole[:, :, self.trunc_size:]
                pitch = target_pitch_whole[:, self.trunc_size:]

            # positonal encoding for attention
            with torch.no_grad():
                attention_ref = attention_ref[:, :, :dec_input_shift.size(1)]
                att_cumsum = att_cumsum[:, :, :dec_input_shift.size(1)]
                
                att_position = att_cumsum / torch.clamp(att_cumsum_whole[:, :, -1:], min=1) * attention_ref
                att_position = torch.sum(att_position, dim=1).unsqueeze(2)
        
            dec_out_dict = self.decoder(
                enc_with_gst,
                dec_input_shift,
                spkr_vec,
                debug,
                attention_ref=attention_ref,
                att_position=att_position,
                pitch=pitch,
                lang_vec=lang_vec
            )
            output_dec = dec_out_dict.get('output_dec')

            # teacher-forcing with tacotron output
            if self.aug_teacher_forcing:
                if self.is_reset:
                    dec_pred_shift = torch.cat([dec_input_padding, output_dec[:, :-1]], dim=1)
                else:
                    dec_pred_shift = torch.cat([dec_input_shift[:, 0:1], output_dec[:, :-1]], dim=1)
                dec_out_dict2 = self.decoder(
                    enc_with_gst, 
                    dec_pred_shift.detach(), 
                    spkr_vec,
                    debug,
                    attention_ref=attention_ref,
                    att_position=att_position,
                    pitch=pitch,
                    is_aug=True,
                    lang_vec=lang_vec
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
                    curr_dec_input = torch.zeros(N, 1, 120).to(enc_with_gst)
                else:
                    curr_dec_input = prev_dec_output[:, -1:]

                curr_att_weight = attention_ref[:, :, i:i+1]
                curr_att_position = att_position[:, i:i+1]
                dec_out_dict = self.decoder(
                    enc_with_gst,
                    curr_dec_input,
                    spkr_vec,
                    debug,
                    attention_ref=curr_att_weight,
                    att_position=curr_att_position,
                    pitch=target_pitch_whole[:, i:i+1],
                    lang_vec=lang_vec
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
            spkr_adv_loss = 0                
            taco_out_dict.update({"spkr_adv_loss": spkr_adv_loss})
            taco_out_dict.update({"spkr_vec": spkr_vec})

            # prenet_loss
            taco_out_dict.update({"prenet_loss": prenet_loss})

            # aligner_loss
            taco_out_dict.update({"att_loss": att_loss})

            # duration_loss
            duration_gt = torch.sum(attention_ref_whole, -1)
            duration_loss = torch.sum((duration_pred - duration_gt)**2) / torch.sum(text_lengths) * 0.01
            taco_out_dict.update({"durpred_loss": duration_loss})

        # update output dictionary
        self.att_weights = attention_ref                    # N x T_enc x T_dec
        taco_out_dict.update({
            'output_dec': output_dec,
            'output_post': output_post,
            'gst_vec': gst_vec,
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
        if 'l' in l:
            print('freeze language embedding.')
            freeze_params(self.lang_embed)
        if 'e' in l:
            print('freeze encoder.')
            freeze_params(self.encoder)
        if 'atte' in l:
            print('freeze att_encoder.')
            freeze_params(self.att_encoder)
        if 'a' in l:
            print('freeze attention.')
            freeze_params(self.attention)
        if 'dp' in l:
            print('freeze duration_predictor.')
            freeze_params(self.duration_predictor)
        if 'd' in l:
            print('freeze decoder.')
            freeze_params(self.decoder)
        if 'p' in l:
            print('freeze post processor.')
            freeze_params(self.post_processor)
        if 'adv_lcls' in l:
            print('freeze adv_lang_classifier.')
            freeze_params(self.adv_lang_classifier)
        if 'scls' in l:
            print('freeze speaker classifier.')
            freeze_params(self.spkr_classifier)


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

    def get_seq_mask(self, batch_size, text_lengths, spec_lengths, whole_spec_lengths=None, device=None):            
        with torch.no_grad():
            text_lengths = torch.tensor(text_lengths, device=device)
            spec_lengths = torch.tensor(spec_lengths, device=device)
            if whole_spec_lengths is None:
                whole_spec_lengths = spec_lengths
            else:
                whole_spec_lengths = whole_spec_lengths

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
        _, enc_output, enc_h_out = self.att_encoder(
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
