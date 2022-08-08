# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import argparse, multiprocessing, torch, yaml, os, sys, shutil, pickle
import numpy as np

from sklearn import mixture, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
from tts_text_util.get_vocab import get_sg_vocab
import voxa
from voxa import VoxaConfig
from voxa.database.data_group import expand_data_group
from voxa.prep.signal_processing import SignalProcessing

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.loader import DataLoader
from tacotron.util.default_args import load_and_override_args, get_default_synthesis_args
from tacotron.synthesize import write_wav
from tacotron.util.gen_common import load_model_common, plot_attention, plot_spec, trim_and_add_silence, auto_clamp, pad_text

def main():
    args = get_default_synthesis_args()

    checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
    ckpt_args = load_and_override_args(checkpoint['args'], checkpoint['args'])

    # manually set ckpt_args to suitable for prosody extraction
    ckpt_args.shuffle_data = 0  # no shuffling to keep order of audio files same as the metadata
    ckpt_args.batch_size = 1
    ckpt_args.pretraining = 0
    ckpt_args.balanced_spkr = 0
    ckpt_args.trunc_size = ckpt_args.spec_limit

    if ckpt_args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron
    elif ckpt_args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron
        
    if args.gpu is None:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        torch.cuda.set_device(args.gpu[0])

    vocab, idx_to_vocab = get_sg_vocab()
    model = Tacotron(ckpt_args, vocab=vocab, idx_to_vocab=idx_to_vocab, transfer=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.reset_decoder_states(debug=ckpt_args.debug)
    model = model.eval()
    print('loaded checkpoint %s' % (args.init_from))

    if args.use_gpu:
        model = model.cuda()

    voxa_config = VoxaConfig.load_from_config(checkpoint['config'], initialize=True)
    sp = SignalProcessing(voxa_config)
    speaker_list = set(ckpt_args.data.split(','))
    target_list = set(expand_data_group(args.gst_spkr.split(','), voxa_config))

    target_list = list(target_list)
    speaker_list = list(speaker_list)

    loader = DataLoader(
        ckpt_args,
        target_list,
        voxa_config,
        speaker_list=speaker_list,
        sort=False,
    )

    # Directory Setting 
    datasetname = f'{args.gst_spkr}_2_{args.tgt_spkr}'  
    args.dir_bin = os.path.join(voxa_config.hdd_bin_root, f'{datasetname}/bin', voxa_config.alias)
    print(f'Data binary saved at {args.dir_bin}')

    dir_bin = {}
    with open(voxa_config.dataset_list_file, 'r') as rFile:
        for line in rFile.readlines():
            dataset, _, curr_dir_bin = line.strip().split('|')
            dir_bin[dataset] = os.path.join(voxa_config.hdd_bin_root, curr_dir_bin, voxa_config.alias)
    load_list_file_path = os.path.join(dir_bin[args.gst_spkr], voxa_config.load_dict_file)
    load_list_tmp = torch.load(load_list_file_path)

    os.makedirs(args.dir_bin, exist_ok=True)
    os.chmod(args.dir_bin, 0o777)

    load_dict_filename = os.path.join(args.dir_bin, voxa_config.load_dict_file)
    bin_file_mel_path = os.path.join(args.dir_bin, voxa_config.mel_spec_binary)
    bin_file_mel = open(bin_file_mel_path, 'wb')
    if hasattr(voxa_config, 'pitch_binary'):
        bin_file_pitch_path = os.path.join(args.dir_bin, voxa_config.pitch_binary)
        bin_file_pitch = open(bin_file_pitch_path, 'wb')
    
    # phoneme binary (cp from gst_spkr)
    shutil.copy(os.path.join(dir_bin[args.gst_spkr], voxa_config.phoneme_sg_binary), os.path.join(args.dir_bin, voxa_config.phoneme_sg_binary))

    offset_mel = 0
    offset_lin = 0
    offset_pitch = 0

    # make file load list
    line_load_dict = load_list_tmp

    print('Voice Conversion...')
    with torch.no_grad():
        for i in tqdm(range(loader.split_sizes['whole'])):
            loader_dict = loader.next_batch('whole')
            phoneme_input = loader_dict.get("x_phoneme")
            target_mel = loader_dict.get("y_specM")
            target_mel_whole = loader_dict.get("y_specM_whole")
            target_pitch = loader_dict.get("y_pitch")
            target_pitch_whole = loader_dict.get("y_pitch_whole")

            subbatch_spec_lengths = loader_dict.get("subbatch_len_spec")
            phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
            spec_lengths = loader_dict.get("len_spec")
            len_mask = loader_dict.get("len_spec_mask")

            spkr_id_src = loader_dict.get("idx_speaker") #referece speaker 
            spkr_id = torch.Tensor([loader.speaker_manager.get_compact_id(args.tgt_spkr)]).type_as(spkr_id_src).cuda()

            spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S

            model_out_dict = model(
                phoneme_input,
                target_mel,
                spkr_id,
                spec_lengths,
                phoneme_lengths,
                gst_source='ref_wav', 
                gst_spkr=spkr_id_src,
                target_mel_whole=target_mel_whole,
                whole_spec_len=spec_lengths,
                debug=ckpt_args.debug,
                exp_no=ckpt_args.exp_no,
                target_pitch=target_pitch,
                target_pitch_whole=target_pitch_whole,
            )
            output_post = model_out_dict.get('output_post')[0]
            seq_end = model_out_dict.get('seq_end').tolist()[0]
            output_post = auto_clamp(output_post, sp, ckpt_args.dec_out_type).cpu().numpy()

            # write tmp wavfile 
            speaker_oid = loader.speaker_manager.get_original_id(spkr_id[0])
            spec_clipped = trim_and_add_silence(output_post[:seq_end])
            write_wav(spec_clipped, speaker_oid, 'tmp.wav', args.spec_pow, sp, ckpt_args.dec_out_type, 
                      args.num_recon_iters, enable_after_effect=args.enable_after_effect,
                      enable_loudness=args.enable_loudness, model_version=ckpt_args.model_version)

            # write mel binary
            # zero-padded audio Tensor
            audio_l_padded, _ = sp.file2padded_waveform('tmp.wav')

            # STFT
            stft = sp.stft(audio_l_padded, output_type='complex')

            # mel-spec
            magnitude = torch.norm(stft, 2, -1)
            mel_spec = sp.spectrogram(magnitude, type='mel', input_type='magnitude', is_numpy=True)
            serialized_mel = pickle.dumps(mel_spec, protocol=pickle.HIGHEST_PROTOCOL)
            bin_file_mel.write(serialized_mel)

            line_load_dict['data'][i]['len_spec'] = len(mel_spec)
            line_load_dict['data'][i]['offset_mel'] = offset_mel
            line_load_dict['data'][i]['read_n_mel'] = len(serialized_mel)
            offset_mel += len(serialized_mel)

            # write pitch binary
            if hasattr(voxa_config, 'pitch_binary'):
                pitch = sp.estimate_pitch(audio_l_padded, use_cuda=args.use_gpu).numpy()
                assert mel_spec.shape[0] == mel_spec.shape[0]
                serialized_pitch = pickle.dumps(pitch, protocol=pickle.HIGHEST_PROTOCOL)
                bin_file_pitch.write(serialized_pitch)

                line_load_dict['data'][i]['offset_pitch'] = offset_pitch
                line_load_dict['data'][i]['read_n_pitch'] = len(serialized_pitch)
                offset_pitch += len(serialized_pitch)

    bin_file_mel.close()
    os.chmod(bin_file_mel_path, 0o777)
    if hasattr(voxa_config, 'pitch_binary'):
        bin_file_pitch.close()
        os.chmod(bin_file_pitch_path, 0o777)

    torch.save(line_load_dict, load_dict_filename)

if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
