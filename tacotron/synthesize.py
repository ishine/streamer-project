# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, json
from collections import namedtuple

import librosa, torch
import numpy as np
from scipy.io import wavfile
from tts_text_util.get_vocab import get_sg_vocab, phoneme2lang, others, symbols
from voxa.prep.signal_processing import SignalProcessing

from tacotron.util.gen_common import gen_output_full_path, gen_prosody_output_full_path, preprocess_audio, plot_attention, \
    load_model_common, trim_and_add_silence, auto_clamp, pad_text, get_pitch_from_audio
from tacotron.util.default_args import get_default_synthesis_args

_effects = {}
eng_vowels = set([
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2',
    'AY0', 'AY1', 'AY2', 'EH0', 'EH1', 'EH2', 'ER0', 'EY0', 'EY1', 'EY2', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
    'IY2', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
])
kor_vowels = set([
    'A_ko', 'o_ko', 'O_ko', 'U_ko', 'u_ko', 'E_ko', 'a_ko', 'e_ko', '1_ko', '2_ko', '3_ko', '4_ko', 
    '5_ko', '6_ko', '7_ko', '8_ko', '9_ko', '[_ko', ']_ko', '<_ko', '>_ko'
])
all_vowels = kor_vowels | eng_vowels

SeqProsodyModel = namedtuple('SeqProsodyModel', 'model args style_list voxa_config data_list speaker_manager lang_dict')

def default_args():
    d = {}
    for arg in get_default_synthesis_args(parse=False):
        d[arg['name']] = arg['default']
    return d


def load_model(new_args, **kwargs):
    if new_args.prosody_source == 'ref_wav':
        ignore_attention = False
    else:
        ignore_attention = True
    loaded = load_model_common(new_args, ignore_attention, **kwargs)
    style_list = loaded.checkpoint.get('cluster_list', None)        
    return SeqProsodyModel(
        loaded.model,
        loaded.args,
        style_list,
        loaded.voxa_config,
        loaded.data_list,
        loaded.speaker_manager,
        loaded.lang_dict
    )
  

def get_speaker_data_list(args):
    return args.data.split(',')


def apply_effect(actor, wave, signal_processing):
    effects = fetch_effects(signal_processing)
    if 'santa' == actor:
        choice = np.random.choice(len(effects['santa']))
        wave = np.concatenate([wave, effects['santa'][choice]], axis=0)
        return wave
    return wave


def fetch_effects(signal_processing):
    if _effects:
        return _effects

    _effects['santa'] = []
    scale = float(os.environ.get('AUDIO_SCALE', 0.8))
    for r, _, fs in os.walk('/data/audio_effects/santa'):
        for f in fs:
            if f.endswith('.wav'):
                wave, _ = librosa.load(os.path.join(r, f), sr=16000)
                wave = signal_processing.loudness(wave, 16000)
                wave = signal_processing.max_norm_s16(wave, scale)
                _effects['santa'].append(wave)
    return _effects


def _gather_gst_from_source(tgt_spkr_id, args, style_list, signal_processing):
    if args.gst_source == 'cluster':
        gst_idx = args.gst_idx

        # will fall back to 'gst_mean' mode if an error occurs.
        if style_list is None:
            args.gst_source = 'gst_mean'
            print(f'Style vectors are not clustered yet. Falling back to gst_mean mode.')
        else:
            style_vector_list = style_list[tgt_spkr_id]
            if gst_idx < 0 or len(style_list[tgt_spkr_id]) < gst_idx:
                args.gst_source = 'gst_mean'
                style_max_index = len(style_vector_list) - 1
                print(f'Style index out of range, valid between 0-{style_max_index}. Falling back to gst_mean mode.')
            else:
                gst_vector = torch.from_numpy(style_vector_list[gst_idx]).float()
                return gst_vector, None

    elif args.gst_source == 'ref_wav':
        ref_mel = None
        if args.gst_ref_wav != '':
            ref_mel = preprocess_audio(args.gst_ref_wav, signal_processing)
        return None, ref_mel

    # use `if` to catch fall-backs to 'gst_mean' mode.
    if args.gst_source == 'gst_mean':
        return None, None

    else:
        raise RuntimeError('choose the style source')
    return _effects


def _gather_prosody_from_source(args, sp):
    if args.prosody_source == 'prosody_vec':
        prosody = []
        if hasattr(args.prosody_ref_file, 'read'):
            json_dict = json.load(args.prosody_ref_file)
        else:
            with open(args.prosody_ref_file, 'r') as json_prosody:
                json_dict = json.load(json_prosody)
        
        for feature in json_dict['features']:
            prosody.append(feature['data'])
        prosody_vector = torch.tensor(prosody)
        prosody_vector = prosody_vector.transpose(0, 1).unsqueeze(0)
        duration_vector = torch.tensor(json_dict['duration']['data'])
        return {
            "prosody_vector": prosody_vector,
            "duration": duration_vector,
        }
    else:
        return {
            "prosody_vector": None,
            "duration": None,
        }


def infer_one(model, text_processed, args, style_list, voxa_config, speaker_manager, data_list=[], lang_dict=None, stop_type=None, is_plot_attention=False, save_prosody=False):
    output = {}
    output['dec_out_type'] = args.dec_out_type
    output['model_version'] = args.model_version

    # set stop type from args if not provided explicitly.
    if stop_type is None:
        stop_type = args.stop_type

    sp = SignalProcessing(voxa_config)

    cleansed = text_processed.text
    enc_input = text_processed.enc_text
    vocab, idx_to_vocab = get_sg_vocab()
    accents = text_processed.accents

    lang_id = None
    if args.fluency == 1 and lang_dict is not None:
        p2l = [phoneme2lang[idx_to_vocab[p]] for p in enc_input if idx_to_vocab[p] not in others+symbols]
        lang = max(set(p2l), key=p2l.count) if len(p2l) > 0 else 'ENG'
        lang_id = torch.LongTensor([lang_dict[lang]])

    # filter out invalid vocab from enc_input and make warning.
    if None in enc_input:
        enc_input = [x for x in enc_input if x is not None]
        output['error'] = {
            "module": 'texa',
            "message": "Detected invalid vocab."
        }

    with torch.no_grad():
        # input text setting
        enc_input, commas = pad_text(enc_input, model_version=args.model_version, debug=args.debug)
        text_lengths = [len(enc_input)]
        enc_input = torch.LongTensor(enc_input).view(1,-1)

        # input accent setting
        if accents is not None:
            accents, _ = pad_text(accents, null_padding=0)
            accents = torch.FloatTensor(accents).view(1,-1)

        # target speaker setting
        tgt_spkr_id = speaker_manager.get_compact_id(args.tgt_spkr)
        tgt_spkr = torch.LongTensor([tgt_spkr_id])

        if args.morph_ratio:
            morph_ratio = [float(m) for m in args.morph_ratio.split(',')]
            total = sum(morph_ratio)
            if total != 1.0:
                morph_ratio = [m / total for m in morph_ratio]
        else:
            morph_ratio = None

        # gst speaker setting
        if args.gst_spkr != '' and args.gst_spkr in data_list:
            if args.gst_spkr in data_list:
                gst_spkr_id = speaker_manager.get_compact_id(args.gst_spkr)
            else:
                print(f"{args.gst_spkr} does not exist. Generate with target spkr's gst.")
                gst_spkr_id = tgt_spkr_id
        else:
            gst_spkr_id = tgt_spkr_id
        gst_spkr = torch.LongTensor([gst_spkr_id])
        gst_vec, ref_mel = _gather_gst_from_source(gst_spkr_id, args, style_list, sp)

        # fine-grained prosody speaker setting
        if args.gst == 0:
            if args.prosody_spkr != '' and args.prosody_spkr in data_list:
                prosody_spkr_id = speaker_manager.get_compact_id(args.prosody_spkr)
            else:
                if not args.prosody_spkr in data_list:
                    print(f"{args.prosody_spkr} does not exist. Generate with target spkr's fine-grained prosody.")
                prosody_spkr_id = tgt_spkr_id
            prosody_spkr = torch.LongTensor([prosody_spkr_id])
            prosody_vec = _gather_prosody_from_source(args, sp)
        elif args.gst == 1:
            # cannot handle fine-grained prosody when gst == 1
            prosody_vec = prosody_spkr = None
        elif args.gst == 2:
            if args.prosody_spkr != '' and args.prosody_spkr in data_list:
                prosody_spkr_id = speaker_manager.get_compact_id(args.prosody_spkr)
            else:
                if not args.prosody_spkr in data_list:
                    print(f"{args.prosody_spkr} does not exist. Generate with target spkr's fine-grained prosody.")
                prosody_spkr_id = tgt_spkr_id
            prosody_spkr = torch.LongTensor([prosody_spkr_id])
            prosody_vec = _gather_prosody_from_source(args, sp)

        # generation length setting
        assert args.wave_limit > 0
        if not ref_mel is None and args.wave_limit == 1000:
            # pad ref_mel to be compatible with r_factor
            if ref_mel.size(1) % args.r_factor != 0:
                N, T, H = ref_mel.size()
                T_pad = args.r_factor - (T % args.r_factor)
                ref_mel = torch.cat([ref_mel, ref_mel.new().resize_(N, T_pad, H)], dim=1)

            if ref_mel.size(1) > args.wave_limit:
                spec_lengths = [int(ref_mel.size(1) * 1.5)]
            else:
                spec_lengths = [args.wave_limit]
            whole_spec_len = torch.LongTensor([ref_mel.size(1)])
        else:
            spec_lengths = [args.wave_limit]
            whole_spec_len = torch.LongTensor(spec_lengths)

        if not args.speed:
            speed = args.speed
        else:
            speed = None

        # gpu setting
        if args.gpu:
            enc_input = enc_input.cuda(1)
            tgt_spkr = tgt_spkr.cuda(1)
            if lang_id is not None:
                lang_id = lang_id.cuda(1)
        if morph_ratio:
            morph_spkrs_ids = [speaker_manager.get_compact_id(f) for f in args.morph_spkrs.split(',')]  # 하나여야만 함
            morph_spkrs = torch.LongTensor(morph_spkrs_ids)
            if args.gpu:
                morph_spkrs = morph_spkrs.cuda(1)

            assert morph_spkrs.size(0) == len(morph_ratio)
            assert len(morph_spkrs) == len(morph_ratio)

            spkr_vec = model.get_mixed_speaker_vector(morph_spkrs, morph_ratio)
            speed = model.get_mixed_speed_vector(morph_spkrs, morph_ratio)
            gst_vec = model.get_mixed_gst_vec(morph_spkrs, morph_ratio, enc_input)
            gst_spkr = None
        else:
            spkr_vec = None

        if args.gpu:
            if not ref_mel is None:
                ref_mel = ref_mel.cuda(1)
            if not gst_spkr is None:
                gst_spkr = gst_spkr.cuda(1)
            if not gst_vec is None:
                gst_vec = gst_vec.cuda(1)
            if not prosody_spkr is None:
                prosody_spkr = prosody_spkr.cuda(1)
            if not prosody_vec is None:
                prosody_vec = prosody_vec.cuda(1)
            whole_spec_len = whole_spec_len.cuda(1)
            if not accents is None:
                accents = accents.cuda(1)

        # spectrogram generation
        model.reset_decoder_states(debug=args.debug)
        if args.gst == 0:
            model_out_dict = model(enc_input, None, tgt_spkr, spec_lengths, text_lengths, debug=args.debug, spkr_vec=spkr_vec,
                                   gst_vec=gst_vec, gst_source=args.gst_source, gst_spkr=gst_spkr,
                                   prosody_vec=prosody_vec, prosody_source=args.prosody_source, prosody_spkr=prosody_spkr, 
                                   stop_type=stop_type, speed=speed, speed_x=args.speed_x,
                                   whole_spec_len=whole_spec_len, target_mel_whole=ref_mel, accents=accents, lang_id=lang_id)
        elif args.gst == 1:
            model_out_dict = model(enc_input, ref_mel, tgt_spkr, spec_lengths, text_lengths, debug=args.debug, spkr_vec=spkr_vec,
                                   gst_vec=gst_vec, gst_source=args.gst_source, gst_spkr=gst_spkr,
                                   stop_type=stop_type, speed=speed, speed_x=args.speed_x, comma_input=commas,
                                   whole_spec_len=whole_spec_len, target_mel_whole=ref_mel,
                                   last_pitch_level=args.last_pitch_level, vowels=all_vowels, accents=accents, lang_id=lang_id)
        elif args.gst == 2:
            model_out_dict = model(enc_input, None, tgt_spkr, spec_lengths, text_lengths, debug=args.debug, spkr_vec=spkr_vec,
                                   gst_vec=gst_vec, gst_source=args.gst_source, gst_spkr=gst_spkr,
                                   prosody_vec=prosody_vec, prosody_source=args.prosody_source, prosody_spkr=prosody_spkr, 
                                   stop_type=stop_type, speed=speed, speed_x=args.speed_x,
                                   whole_spec_len=whole_spec_len, target_mel_whole=ref_mel, accents=accents, lang_id=lang_id)

        # dec_out_type can be ('lin', 'mel') and target_type can be ('lin', 'mel', 'wav')
        if args.dec_out_type == 'lin' and args.target_type == 'mel':
            pred_spec = model_out_dict.get('output_dec')
        elif args.dec_out_type == 'mel' and args.target_type == 'lin':
            raise RuntimeError(
                'Linear output is not supported for this model. Bypass this using wave target type.')
        else:
            pred_spec = model_out_dict.get('output_post')
        seq_end = model_out_dict.get('seq_end')
        prosody = model_out_dict.get('prosody_vec')
        prosody_att = model_out_dict.get('prosody_att')

        # write output files
        pred_spec = auto_clamp(pred_spec, sp, args.dec_out_type)
        pred_spec = pred_spec.data.cpu().numpy()
        outpath1, outpath2, outpath3 = gen_output_full_path(args, cleansed)
        spec_clipped = trim_and_add_silence(pred_spec[0, :seq_end[0]])
        if args.no_vocoder == 1 or args.target_type != 'wav':
            write_spec(spec_clipped, outpath3)
        else:
            wave = write_wav(spec_clipped, args.tgt_spkr, outpath1, args.spec_pow, sp, args.dec_out_type,
                      args.num_recon_iters, enable_after_effect=args.enable_after_effect,
                      enable_loudness=args.enable_loudness, model_version=args.model_version)
            return wave

        # plot spectrogram
        if is_plot_attention:
            if type(model.att_weights) is list:
                attention = torch.cat(model.att_weights, dim=-1).squeeze()
            else:
                attention = model.att_weights.squeeze()
                
            plot_attention(None, attention, outpath2)

        # save prosodic features
        if prosody is not None and save_prosody:
            n = 0
            prosody_out1, prosody_out2 = gen_prosody_output_full_path(args, n)
            if prosody_att is not None:
                plot_attention(None, prosody_att.squeeze(0), prosody_out1)

            json_dict = dict()
            json_dict['phoneme_seq'] = [idx_to_vocab[key] for key in enc_input[n].tolist()]
            json_dict['phoneme_location'] = [i for i in range(len(json_dict['phoneme_seq']))]
            json_dict['features'] = []
            for m in range(prosody.size(-1)):
                prosody_dict = {'title': m, 'data': prosody[n, :, m].tolist()}
                json_dict['features'].append(prosody_dict)

            if hasattr(prosody_out2, 'write'):
                json.dump(json_dict, prosody_out2)
            else:
                with open(prosody_out2, 'w') as f:
                    json.dump(json_dict, f)

    return output


def write_wav(spectrogram, speaker_oid, file_path, spec_pow, signal_processing, dec_out_type, num_iters, **kwargs):
    enable_after_effect = kwargs.get('enable_after_effect')
    enable_loudness = kwargs.get('enable_loudness')
    model_version = kwargs.get('model_version')

    if signal_processing.mel_dim == 120 and signal_processing.sample_rate == 16000:
        if dec_out_type == 'lin':
            spectrogram = signal_processing.lin2mel(spectrogram)
        if model_version is not None:
            if model_version == 'cats3' or model_version == 'cats4':
                wave, sr = signal_processing.regan(spectrogram, '210921_repfb2-wo_preemphasis')
            else:
                wave, sr = signal_processing.regan(spectrogram, 'etri_F')
        else:
            wave, sr = signal_processing.regan(spectrogram, 'etri_F')
    else:
        wave = signal_processing.spectrogram2wav(spectrogram,
                                                    num_iters=num_iters,
                                                    spec_pow=spec_pow,
                                                    enable_loudness=enable_loudness
                                                    )
        sr = signal_processing.sample_rate

    if enable_after_effect:
        wave = apply_effect(speaker_oid, wave, signal_processing)

    return wave

    wavfile.write(file_path, sr, wave)

def write_spec(spectrogram, file_path):
    np.save(file_path, spectrogram)
