# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

# import from outside of the project
import sys
# sys.path.insert(1, '/nas/yg/project/WaveRNN_deploy')        # insert at 1, 0 is the script path (or '' in REPL)
from torch.multiprocessing import Process, set_start_method

set_start_method('spawn', force="True")

import numpy as np
import torch, multiprocessing, os.path

import tts_text_util as texa
from scipy.io import wavfile
from tts_text_util.process_text_input import TextToInputSequence
from voxa.prep.signal_processing import SignalProcessing
from voxa.speakermgr.speaker_manager import SpeakerManager

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.util.default_args import get_default_bulkconv_args
from tacotron.util.gen_common import get_args_checkpoint, load_tacotron, set_voxa_config, trim_and_add_silence, auto_clamp, pad_text

def main():
    new_args = get_default_bulkconv_args()
    n_gpu = len(new_args.gpu)
    batch_size = new_args.batch_size
    lang = new_args.lang
    use_gpu = None

    if n_gpu > 0:
        use_gpu = True
        n_worker = n_gpu
        if new_args.n_cpu != 1:
            print('n_cpu is ignored in gpu mode.')
    else:   
        use_gpu = False
        n_worker = max(1, new_args.n_cpu)

    args, checkpoint = get_args_checkpoint(new_args)

    if args.gen_idx < 1:
        raise RuntimeError("Please specify gen_idx. This will be reflected in name of the generated files.")

    os.makedirs(args.out_dir, exist_ok=True)                # make output directory if not exists.

    # print args of current experiment
    for key in args.__dict__:
        print(key, args.__dict__[key])

    # sort sentences to generate by length in decreasing order
    meta_path = args.meta_from
    meta_list = []
    batch_list = []
    with open(meta_path, 'r') as rFile:
        # example of line: "20681|seokhun|아무것도 필요하지 않다.|0.9"
        lines = [x.strip().split('|') for x in rFile.readlines()]
    sorted_lines = sorted(lines, key=lambda x: len(x[2]), reverse=True)

    # make batch
    min_batch_size = batch_size
    max_text_len = curr_max_text_len = len(sorted_lines[0][2])
    for line_split in sorted_lines:
        if args.adaptive_batch == 1 and len(batch_list) == 0:
            curr_max_text_len = len(line_split[2])
            batch_size = max(1, int(min_batch_size * max_text_len / curr_max_text_len))

        # skip sfx
        if line_split[1] == "sfx":
            continue
        else:
            batch_list.append(line_split)

        if len(batch_list) == batch_size:
            meta_list.append(batch_list)
            batch_list = []

    if len(batch_list) > 0:
        meta_list.append(batch_list)
        
    # multi-process
    processes = []
    metalines_per_worker = [meta_list[i::n_worker] for i in range(n_worker)]
    for idx in range(n_worker):
        if len(metalines_per_worker[idx]) >= 1:
            if use_gpu:
                device = new_args.gpu[idx]
            else:
                device = 'cpu'

            process = Process(target=batch_inference, args=(device, metalines_per_worker[idx], new_args, lang))
            process.start()
            processes.append(process)

    for p in processes:
        p.join()


def batch_inference(gpu, meta_list, new_args, lang):
    device = torch.device(gpu)
    args, checkpoint = get_args_checkpoint(new_args)
    data_list = sorted(args.data.split(','))
    speaker_manager = SpeakerManager(data_list)
    voxa_config = set_voxa_config(new_args, checkpoint)
    sp = SignalProcessing(voxa_config)

    gst_vectors = []
    # fall back to gst_mean mode if no style is available.
    mean_styles = checkpoint['state_dict']['prosody_stats.means']
    for i in range(mean_styles.size(0)):
        gst_vectors.append(mean_styles[i].unsqueeze(0))

    style_list = checkpoint.get('cluster_list', None)
    if style_list is not None and args.no_style == 0:
        for i, l in enumerate(style_list):
            if len(l) > 0:
                gst_vectors[i] = torch.stack([torch.from_numpy(a).float() for a in l], dim=0)

    # If target_type is not specified, automatically decide target_type
    if args.target_type == 'auto':
        args.target_type = args.dec_out_type

    model = load_tacotron(args, checkpoint).to(device)
    phoneme_util = TextToInputSequence(True, use_sg=args.use_sg, use_pause=True)

    # manually set offset of expected_sequence_end. (for iyuno typecast studio)
    model.set_manual_seqend_offset(args.seqend_offset)

    # use half precision
    if args.half == '1':
        model = model.half()

    print(f"using device: {device}")
    with torch.no_grad():
        for batch_list in meta_list:
            file_path_list = []
            spkr_vec_list = []
            phoneme_list = []
            phoneme_lengths = []
            spec_lengths = []
            gst_batch_list = []

            max_phoneme_len = -1
            curr_batch_size = 0
            for l in batch_list:
                file_prefix = args.out_dir + "/" + l[0]
                tgt_spkr_id = speaker_manager.get_compact_id(l[1])
                if args.convert_eng2kor == 0:
                    curr_phoneme_input = phoneme_util.get_input_sequence(l[2], lang, convert_eng2kor=False)['index_sequence']
                else:
                    curr_phoneme_input = phoneme_util.get_input_sequence(l[2], lang, convert_eng2kor=True)['index_sequence']
                curr_phoneme_input = pad_text(curr_phoneme_input, null_padding=120)
                phoneme_len = len(curr_phoneme_input)
                curr_gst_vector_batch = gst_vectors[tgt_spkr_id]

                max_phoneme_len = max(phoneme_len, max_phoneme_len)
                N = curr_gst_vector_batch.size(0)
                if N >= 100:
                    raise RuntimeError("Current gen_idx can represent maximum number of style 100.")

                curr_batch_size += N
                file_path_list += [f"{file_prefix}_gst{args.gen_idx * 100 + n}.{args.target_type}" for n in range(N)]
                spkr_vec_list += [tgt_spkr_id for _ in range(N)]
                phoneme_lengths += [phoneme_len for _ in range(N)]
                spec_lengths += [args.spec_limit for _ in range(N)]
                phoneme_list += [torch.LongTensor(curr_phoneme_input).expand(N, -1)]
                gst_batch_list += [curr_gst_vector_batch]

            phoneme_input = torch.LongTensor(curr_batch_size, max_phoneme_len).zero_()
            cursor = 0
            for p in phoneme_list:
                N, T = p.size()
                phoneme_input[cursor:cursor+N, :T].copy_(p)
                cursor = cursor + N
            spkr_id = torch.LongTensor(spkr_vec_list)
            gst_vector_batch = torch.cat(gst_batch_list, dim=0)

            if args.gpu:
                phoneme_input = phoneme_input.to(device)
                spkr_id = spkr_id.to(device)
                gst_vector_batch = gst_vector_batch.to(device)

            model.reset_decoder_states()
            if args.gst == 0:
                model_out_dict = model(phoneme_input, None, spkr_id, spec_lengths, phoneme_lengths, debug=args.debug,
                                    gst_vec=gst_vector_batch, gst_source='cluster', prosody_source='prediction',
                                    stop_type=args.stop_type, speed_x=args.speed_x)
            elif args.gst == 1:
                model_out_dict = model(phoneme_input, None, spkr_id, spec_lengths, phoneme_lengths, debug=args.debug,
                                    gst_vec=gst_vector_batch, gst_source='cluster',
                                    stop_type=args.stop_type, speed_x=args.speed_x)

            # dec_out_type can be ('lin', 'mel') and target_type can be ('lin', 'mel', 'wav')
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
            for i, file_path in enumerate(file_path_list):
                spec_clipped = trim_and_add_silence(pred_spec[i, :seq_end[i]])
                if args.target_type == 'wav':
                    write_wav(spec_clipped, args.num_recon_iters, file_path, args.spec_pow, sp)
                else:
                    write_spec(spec_clipped, file_path)


def write_wav(spectrogram, num_iters, file_path, spec_pow, signal_processing):
    wave = signal_processing.spectrogram2wav(spectrogram, num_iters=num_iters, spec_pow=spec_pow)
    wavfile.write(file_path, 16000, wave)


def write_spec(spectrogram, file_path):
    np.save(file_path, spectrogram)


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            p.terminate()
