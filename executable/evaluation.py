# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os.path, yaml, sys
import numpy as np
import torch
import tts_text_util as texa
from tts_text_util.process_text_input import TextToInputSequence
from voxa.prep.signal_processing import SignalProcessing

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.util.default_args import get_default_evaluation_args
from tacotron.util.gen_common import load_model_common, plot_attention, plot_spec, trim_and_add_silence, auto_clamp, pad_text

from scipy.io import wavfile


def main():
    new_args = get_default_evaluation_args()

    # For Veripe, load checkpoint automatically if there is only one checkpoint file
    if new_args.init_from == '':
        checkpoint_list = [ckpt for ckpt in os.listdir('./checkpoint/') if ckpt.endswith('.t7')]
        if len(checkpoint_list) == 1:
            new_args.init_from = os.path.join('./checkpoint/', checkpoint_list[0])
        else:
            raise RuntimeError("Need to specify checkpoint.")

    loaded = load_model_common(new_args)
    args = loaded.args
    model = loaded.model
    voxa_config = loaded.voxa_config
    speaker_manager = loaded.speaker_manager
    lang_dict = loaded.lang_dict
    phoneme_util = TextToInputSequence(True, use_sg=args.use_sg)
    remove_dummy_pho = True if args.remove_dummy_pho == 1 else False
    sp = SignalProcessing(voxa_config)

    # setting for server
    if args.server == 1:
        args.gpu = None         # server does not have gpu.
    else:
        os.makedirs(args.out_dir, exist_ok=True)    # make out_dir if not exists.

    mdl_spkr_txt_pairs = yaml.load(open(os.path.join(ROOT_PATH, args.mdl_spkr_txt_pairs_from), 'r'), yaml.SafeLoader)
    default_mst_pairs = mdl_spkr_txt_pairs['default']

    if args.speaker_id_list != '':
        speaker_id_list = sorted(list(set(args.speaker_id_list.split(','))))
    else:
        # retrieve target speaker list for the evaluation.
        whole_eval_speakers = yaml.load(open(os.path.join(ROOT_PATH, args.eval_speakers_from), 'r'), yaml.SafeLoader)
        curr_eval_speaker_group = mdl_spkr_txt_pairs\
            .get(args.eval_set, default_mst_pairs)\
            .get('speaker_group', default_mst_pairs['speaker_group'])
        if type(curr_eval_speaker_group) is str:
            # curr_eval_speaker_group is key if its type is 'str'
            tgt_spkr = whole_eval_speakers[curr_eval_speaker_group]
        elif type(curr_eval_speaker_group) is list:
            # curr_eval_speaker_group itself is tgt_spkr list if its type is 'list'
            tgt_spkr = curr_eval_speaker_group

        # add unique speakers to tgt_spkr if exists
        whole_speakers = whole_eval_speakers['whole_set'] + whole_eval_speakers['exclude_group']
        unique_speaker = set(loaded.data_list) - set(whole_speakers)
        if len(unique_speaker) > 0:
            tgt_spkr += list(unique_speaker)

        # ensure that tgt_spkr has at least one speaker
        if len(tgt_spkr) == 0:
            tgt_spkr.append(loaded.data_list[0])

        speaker_id_list = sorted(list(set(tgt_spkr)))

    cpt_id_list = []
    for spkr in speaker_id_list:
        cpt_id = speaker_manager.get_compact_id(spkr)
        if cpt_id is not None:
            cpt_id_list.append(cpt_id)

    assert len(cpt_id_list) > 0
    spkr_vec_list = torch.LongTensor(cpt_id_list)

    if spkr_vec_list.size(0) < args.batch_size:
        args.batch_size = spkr_vec_list.size(0)

    # retrieve sentence list for the evaluation.
    if args.text_from != '':
        with open(args.text_from, 'r') as rFile:
            sentence_list = rFile.readlines()
        sentence_list = [x.strip().split('|') for x in sentence_list if len(x.strip()) > 0]
    else:
        whole_eval_sentences = yaml.load(open(os.path.join(ROOT_PATH, args.eval_sentences_from), 'r'), yaml.SafeLoader)
        text_group_list = mdl_spkr_txt_pairs\
            .get(args.eval_set, default_mst_pairs)\
            .get('text_group', default_mst_pairs['text_group'])

        sentence_list = []
        for text_group in text_group_list:
            sentence_dict_list = whole_eval_sentences.get(text_group, None)
            if sentence_dict_list is None:
                raise RuntimeError('Error occurred while loading evaluation sentences from `args.eval_sentences_from`')
            for d in sentence_dict_list:
                sentence_list.append((d['text'], d['lang']))

    # print args of current experiment
    for key in args.__dict__:
        print(key, args.__dict__[key])

    with torch.no_grad():
        for sentence_idx, sentence_tuple in enumerate(sentence_list):
            sentence = sentence_tuple[0]
            lang = sentence_tuple[1]
            lang_id = None
            if lang_dict is not None:        
                lang_id = torch.LongTensor([lang_dict[lang]])

            total_iteration = int(np.ceil(spkr_vec_list.size(0) / args.batch_size))
            for i in range(total_iteration):
                if i < total_iteration - 1:
                    N = args.batch_size
                else:
                    if spkr_vec_list.size(0) % args.batch_size != 0:
                        N = spkr_vec_list.size(0) % args.batch_size
                    else:
                        N = args.batch_size

                start_idx = i * args.batch_size

                phoneme_input = phoneme_util.get_input_sequence(sentence, lang, rm_dummy_ph=remove_dummy_pho)['index_sequence']
                phoneme_input, commas = pad_text(phoneme_input, model_version=args.model_version, debug=args.debug)
                phoneme_lengths = [len(phoneme_input) for _ in range(N)]
                phoneme_input = torch.LongTensor(phoneme_input).view(1,-1).expand(N,-1)
                if lang == 'JPN':
                    accents = phoneme_util.get_input_sequence(sentence, lang, rm_dummy_ph=remove_dummy_pho)['accent_index_sequence']
                    accents, _ = pad_text(accents, null_padding=0)
                    accents = torch.FloatTensor(accents).view(1,-1).expand(N,-1)
                else:
                    accents = None            

                spkr_id = spkr_vec_list[start_idx: start_idx + N]
                if args.gpu:
                    phoneme_input = phoneme_input.cuda()
                    spkr_id = spkr_id.cuda()
                    if lang == 'JPN':
                        accents = accents.cuda()
                    if lang_id is not None:
                        lang_id = lang_id.cuda()

                spec_lengths = [args.spec_limit]
                whole_spec_lengths = torch.LongTensor(spec_lengths)
                if args.gpu:
                    whole_spec_lengths = whole_spec_lengths.cuda()

                model.reset_decoder_states()

                if args.gst == 0:
                    model_out_dict = model(phoneme_input, None, spkr_id, spec_lengths, phoneme_lengths, debug=args.debug,
                                        gst_source='gst_mean',  prosody_source='prediction', speed_x=args.speed_x,
                                        stop_type=args.stop_type, whole_spec_len = whole_spec_lengths, accents=accents, lang_id=lang_id)
                elif args.gst == 1:
                    model_out_dict = model(phoneme_input, None, spkr_id, spec_lengths, phoneme_lengths, debug=args.debug,
                                        gst_source='gst_mean',  prosody_source='prediction', speed_x=args.speed_x, comma_input=commas,
                                        stop_type=args.stop_type, whole_spec_len = whole_spec_lengths, accents=accents, lang_id=lang_id)
                elif args.gst == 2:
                    model_out_dict = model(phoneme_input, None, spkr_id, spec_lengths, phoneme_lengths, debug=args.debug,
                                        gst_source='gst_mean',  prosody_source='prediction', speed_x=args.speed_x,
                                        stop_type=args.stop_type, whole_spec_len = whole_spec_lengths, accents=accents, lang_id=lang_id)

                # dec_out_type can be ('lin', 'mel') and target_type can be ('lin', 'mel', 'wav')
                if args.target_type == 'auto':
                    args.target_type = args.dec_out_type
                if args.dec_out_type == 'lin' and args.target_type == 'mel':
                    pred_spec = model_out_dict.get('output_dec')
                elif args.dec_out_type == 'mel' and args.target_type == 'lin':
                    raise RuntimeError(
                        'Linear output is not supported for this model. Bypass this using wave target type.')
                else:
                    pred_spec = model_out_dict.get('output_post')
                seq_end = model_out_dict.get('seq_end').tolist()

                # write output files
                pred_spec = auto_clamp(pred_spec, sp, args.dec_out_type)
                pred_spec = pred_spec.data.cpu().numpy()
                common_path = args.out_dir + '/' + args.exp_no + '_t' + str(sentence_idx) + '_'
                file_path_list = [f'{common_path}{speaker_manager.get_original_id(speaker_id)}.{args.target_type}'
                                  for speaker_id in spkr_vec_list[start_idx: start_idx + N].tolist()]
                for i, file_path in enumerate(file_path_list):
                    spec_clipped = trim_and_add_silence(pred_spec[i, :seq_end[i]])
                    speaker_oid = speaker_manager.get_original_id(spkr_id[i])
                    if args.target_type == 'wav':
                        write_wav(spec_clipped, speaker_oid, file_path, args.spec_pow, sp, args.dec_out_type,
                                  sentence, lang, args.server, args.vocoder_type, model_version=args.model_version)
                    else:
                        write_spec(spec_clipped, file_path, args.server)

                # plot spectrogram
                if args.server == 0:
                    attentions = model.att_weights
                    mel_path_list = [common_path + speaker_manager.get_original_id(speaker_id) + '_mel.png' for speaker_id
                                      in spkr_vec_list[start_idx: start_idx + N].tolist()]
                    att_path_list = [common_path + speaker_manager.get_original_id(speaker_id) + '_att.png' for speaker_id
                                      in spkr_vec_list[start_idx: start_idx + N].tolist()]
                    pred_spec = np.transpose(pred_spec, (0, 2, 1))
                    for j in range(N):
                        plot_spec(pred_spec[j], mel_path_list[j], sp)
                        plot_attention(None, attentions[j, :(seq_end[j] // args.r_factor)], att_path_list[j])


def write_wav(spectrogram, speaker_oid, file_path, spec_pow, signal_processing, dec_out_type, sentence, lang, server, vocoder_type, **kwargs):
    model_version = kwargs.get('model_version')
    if vocoder_type in ['regan', 'wr'] and dec_out_type == 'lin':
        spectrogram = signal_processing.lin2mel(spectrogram, spow=spec_pow)

    if vocoder_type == 'regan':
        if model_version is not None:
            if model_version == 'cats3' or model_version == 'cats4':
                wave, sr = signal_processing.regan(spectrogram, '210921_repfb2-wo_preemphasis')
            else:
                wave, sr = signal_processing.regan(spectrogram, 'etri_F')
        else:
            wave, sr = signal_processing.regan(spectrogram, 'etri_F')
    elif vocoder_type == 'wr':
        wave, sr = signal_processing.typecast_hd1_vocoder(spectrogram, speaker_oid)
    elif vocoder_type == 'gl':
        wave = signal_processing.spectrogram2wav(spectrogram,
                                                 num_iters=50,
                                                 spec_pow=spec_pow,
                                                 enable_loudness=0
                                                 )
        sr = signal_processing.sample_rate
    else:
        raise RuntimeError('Not supported vocoder type.')

    if server == 0:
        wavfile.write(file_path, sr, wave)
    else:
        import eval_util
        lang_dict = {'KOR': 'ko-KR', 'ENG': 'en-US', 'JPN': 'ja-JP', 'ESP': 'es-ES', 'FRA': 'fr-FR'}
        eval_util.write_wav(file_path, sr, wave, sentence, lang_dict[lang])


def write_spec(spectrogram, file_path, server):
    if server == 0:
        np.save(file_path, spectrogram)
    else:
        raise RuntimeError("Mel output is not supported in veripe server.")


if __name__ == '__main__':
    main()
