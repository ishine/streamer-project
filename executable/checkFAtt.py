# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, sys

num_core = os.cpu_count()
if num_core == 20:
    os.putenv("OMP_NUM_THREADS", str(num_core // 4))
else:
    os.putenv("OMP_NUM_THREADS", str(num_core // 2))

import multiprocessing

import numpy as np
import torch
from tts_text_util.get_vocab import get_sg_vocab
import voxa
from voxa.database.data_group import expand_data_group

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.util.default_args import get_default_training_args, get_backward_compatible_args
from tacotron.loader import DataLoader
from tacotron.util.train_common import verify_args
from tacotron.util.gen_common import plot_attention, plot_spec


def main():
    args = get_default_training_args()
    verify_args(args)

    # check if newly added argument supports backward compatibility
    curr_args = get_default_training_args(parse=False)
    backward_args_set = set([x['name'] for x in get_backward_compatible_args(parse=False)])
    missing_args = []
    for d in curr_args:
        arg_name = d['name']
        if arg_name not in backward_args_set:
            missing_args.append(arg_name)
    if len(missing_args) > 0:
        missing_args = ', '.join(missing_args)
        print(f'Must add new arguments in backward_compatible_args: {missing_args}')
        raise RuntimeError()

    # this resolution scale is fixed in the model definition.
    assert args.spec_limit % args.r_factor == 0
    assert args.veripe_every % args.save_every == 0
            
    torch.manual_seed(0)
    np.random.seed(0)
    iteration = 0

    # load checkpoint file and set voxa_config
    speaker_list = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
        ckpt_args = checkpoint['args']
        ckpt_voxa_config = voxa.VoxaConfig.load_from_config(checkpoint['config'], initialize=True)
        voxa_config = ckpt_voxa_config
        voxa_config.refresh_env_variables()
    else:
        raise RuntimeError('Must specify checkpoint path')

    if args.dec_out_type == 'mel':
        # post_output_size must be the same with the decoder_output_size
        args.post_out_size = args.dec_out_size = voxa_config.mel_dim
    else:
        args.post_out_size = int(voxa_config.n_fft // 2) + 1
        args.dec_out_size = voxa_config.mel_dim

    # reuse args in checkpoint to prevent incompatibility
    args.config = ckpt_args.config
    args.use_sg = ckpt_args.use_sg
    args.charvec_dim = ckpt_args.charvec_dim
    args.enc_hidden = ckpt_args.enc_hidden
    args.att_hidden = ckpt_args.att_hidden
    args.dec_hidden = ckpt_args.dec_hidden
    args.dec_out_type = ckpt_args.dec_out_type
    args.spkr_embed_size = ckpt_args.spkr_embed_size
    args.prosody_size = ckpt_args.prosody_size
    args.key_size = ckpt_args.key_size
    args.context_hidden_size = ckpt_args.context_hidden_size
    args.num_trans_layer = ckpt_args.num_trans_layer
    args.r_factor = ckpt_args.r_factor
    args.att_range = ckpt_args.att_range
    args.fluency = ckpt_args.fluency
    args.dropout = ckpt_args.dropout
    args.stop_type = ckpt_args.stop_type
    args.gst = ckpt_args.gst
    args.conv = ckpt_args.conv
    args.debug = ckpt_args.debug
    args.n_token = ckpt_args.n_token
    args.n_head = ckpt_args.n_head
    args.model_version = ckpt_args.model_version
    args.prosody_ignore_spkrs = ckpt_args.prosody_ignore_spkrs

    # set trunc_size to spec_limit (no TBPTT)
    args.trunc_size = args.spec_limit

    # No shuffle and no validataion dataset
    args.batch_size = 1
    args.shuffle = 0
    args.no_validation = 1

    if args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron
    elif args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron

    # set dataset option
    speaker_list = sorted(ckpt_args.data.split(','))
    if args.data is not None:
        dataset_list = sorted(expand_data_group(args.data.split(','), voxa_config))
    else:
        dataset_list = speaker_list

    loader = DataLoader(args, dataset_list, voxa_config, speaker_list=speaker_list, sort=False, ssd=False)
    args.num_id = loader.speaker_manager.get_num_speakers()
    loader_train_flag = 'train'

    # set misc options
    os.makedirs(args.save_dir, exist_ok=True)               # make save directory if not exists.
    vocab, idx_to_vocab = get_sg_vocab()
    args.vocab_size = len(vocab)

    iter_per_epoch = loader.split_num_iters[loader_train_flag]
    if args.max_iters > 0:
        max_iters = args.max_iters + iter_per_epoch - (args.max_iters % iter_per_epoch)
        max_epoch = int(max_iters / iter_per_epoch)
        # if max_iter is given, save at the end of the training.
        if args.print_every == -1:
            args.print_every = int(max_epoch / 20) * iter_per_epoch
        if args.save_every == -1:
            args.save_every = max_epoch * iter_per_epoch
        if args.veripe_every == -1:
            args.veripe_every = max_epoch * iter_per_epoch
    else:
        max_iters = iter_per_epoch * args.max_epochs
        if args.print_every == -1:
            args.print_every = iter_per_epoch
        if args.save_every == -1:
            args.save_every = iter_per_epoch * 10    # save every 10 epoch by default
        if args.veripe_every == -1:
            args.veripe_every = iter_per_epoch * 30    # save every 10 epoch by default

    model = Tacotron(args, vocab=vocab, idx_to_vocab=idx_to_vocab)
    model_optim = None
    state_dict = checkpoint['state_dict']

    # ignore the decoder, predictor in checkpoint
    for key in model.state_dict().keys():
        if key.startswith('duration_predictor') or key.startswith('decoder'):
            state_dict[key] = model.state_dict()[key]
    keys_to_pop = []
    for key in state_dict:
        if not key in model.state_dict().keys():
            keys_to_pop.append(key)
    for key in keys_to_pop:
        state_dict.pop(key)
    model.load_state_dict(state_dict)
    
    # set single/multi gpu usage
    if args.gpu:
        torch.cuda.manual_seed(0)
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model = model.cuda()
    model = model.eval()

    print('Start checking... (1 epoch = %s iters)' % (iter_per_epoch))

    # # for whole speakers
    # whole_list = []
    # while iteration < max_iters + 1:
    #     spkr_id, data_idx, att_nll, _ = one_iteration(model, model_optim, loader, args, iteration, loader_train_flag)

    #     speaker_oid = loader.speaker_manager.get_original_id(spkr_id)
    #     whole_list.append([speaker_oid, data_idx+1, att_nll])

    #     if loader.split_cursor[loader_train_flag] == 0 and loader.is_subbatch_end[loader_train_flag]:
    #         break
    
    # for i, line in enumerate(sorted(whole_list, key=lambda x: x[2], reverse=True)):
    #     if i < args.num_id * 10:
    #         print(' | '.join([str(x) for x in line]))

    # speaker-wise
    whole_dict = {}
    worst_nll = -np.inf
    while iteration < max_iters + 1:
        spkr_id, data_idx, att_nll, attention, tmw, pi, att_key = one_iteration(model, model_optim, loader, args, iteration, loader_train_flag)

        speaker_oid = loader.speaker_manager.get_original_id(spkr_id)
        if not speaker_oid in whole_dict:
            whole_dict[speaker_oid] = []
        whole_dict[speaker_oid].append([data_idx+1, att_nll, attention])

        if att_nll > worst_nll:
            worst_nll = att_nll
            worst_nll_recon_set = [spkr_id, tmw, pi, attention, att_key]
        # if iteration+1 > 20:
        #     worst_nll_recon_set = [spkr_id, tmw, pi, attention, att_key]
        #     break

        if loader.split_cursor[loader_train_flag] == 0 and loader.is_subbatch_end[loader_train_flag]:
            break
            
        iteration += 1

    from scipy.io import wavfile
    from tacotron.util.gen_common import plot_attention
    from tts_text_util.process_text_input import TextToInputSequence
    from tts_text_util.g2p_ko.korean_g2p_code import conv_code_to_hangul_interactive
    from io import BytesIO
    def write_wav(spectrogram, speaker_oid, file_path, spec_pow, signal_processing):
        wave, sr = signal_processing.regan(spectrogram, '210921_repfb2-wo_preemphasis')
        wavfile.write(file_path, sr, wave)
    sp = voxa.prep.signal_processing.SignalProcessing()

    spkr_id, target_mel_whole, phoneme_input, attention, att_key = worst_nll_recon_set
    print(phoneme_input)

    p_cho_list = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"]
    p_jung_list = ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ", "ㅘ", "ㅝ", "ㅚ", "ㅟ", "ㅙ", "ㅞ", "ㅢ", ""]
    p_jong_list = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ", ""]
    p_cho_Roman = ["g", "n", "d", "l", "m", "b", "s", "-", "j", "q", "k", "t", "p", "h", "x", "w", "f", "c", "z"]
    p_jung_Roman = ["A", "o", "O", "U", "u", "E", "a", "e", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", "<", ">", "/"]
    p_jong_Roman = ["G", "N", "D", "L", "M", "B", "0", "="]

    roman_to_hangul = {}
    p_han = p_cho_list+p_jung_list+p_jong_list
    p_rom = p_cho_Roman+p_jung_Roman+p_jong_Roman
    for i in range(len(p_han)):
        roman_to_hangul[p_rom[i]] = p_han[i]
    # print(1-torch.clamp(torch.sum(model.att_weights, dim=1), max=1).cpu())
    # for j in range(args.batch_size):
    #     print((spec_lengths[j]//args.r_factor).item(), torch.sum(torch.gt(model.att_weights[j], 0.5).long()).cpu().item(), torch.sum(model.att_weights[j]).cpu().item())
    for i in range(args.batch_size):
        # # plot_attention(None, model_out_dict.get('prosody_att').detach()[i], f'/nas/yg/data/att{i}-prosody.png')
        # plot_attention(None, model.att_weights.detach()[i], f'/nas/yg/data/att{i}-encdec.png')
        # # plot_attention(None, model.att_weights.detach()[i].transpose(0,1) - model_out_dict.get('prosody_att').detach()[i], f'/nas/yg/data/att{i}-diff.png')
        # write_wav(pred_lin.data.cpu()[i], spkr_id[i].item(), f'/nas/yg/data/att{i}.wav', 1.2, sp)
        write_wav(target_mel_whole.cpu()[i], spkr_id, f'/nas/yg/data/att{i}.wav', 1.2, sp)
        plot_attention(None, attention[0], f'/nas/yg/data/att{i}.png')
        # plot_spec(target_mel_whole.cpu()[i], f'/nas/yg/data/spec{i}_gt.png', sp)
        # plot_spec(torch.bmm(attention.transpose(1,2), att_key)[i], f'/nas/yg/data/spec{i}_pred.png', sp)

        max_val, alignment = torch.max(attention[i], dim=0)
        attended_phoneme_idx = torch.index_select(phoneme_input[i], 0, alignment).tolist()
        phoneme_seq = [idx_to_vocab[x] for x in attended_phoneme_idx]
        han_seq = []
        for x in phoneme_seq:
            q = x.split("_")[0]
            if q in roman_to_hangul:
                han_seq.append(roman_to_hangul[q])
            else:
                han_seq.append(q)
        print(phoneme_seq)
        print(han_seq)

    for j, x in enumerate(phoneme_seq):
    # for j, x in enumerate(han_seq):
        # loss = torch.nn.functional.l1_loss(target_lin[:, j:j+4], pred_lin[:, j:j+4]).item()
        # print(0.05*(j+1),"[]", x, "[]",max_val[j].item(), "[]",loss)
        print(0.0125*(j+1),"[]", x, "[]",max_val[j].item())
    
    for s in sorted(whole_dict.keys()):
        print(''.join(['-' for _ in range(100)]))
        print(f'SPEAKER: {s}')
        for i, line in enumerate(sorted(whole_dict[s], key=lambda x: x[1], reverse=True)):
            if i < 5:
                file_idx = line[0]
                curr_attention = line[-1]
                plot_attention(None, curr_attention[0], f'/nas/yg/data/checkFA-worst_att{file_idx}.png')
            #     write_wav(target_lin.cpu()[i], spkr_id[i].item(), f'/nas/yg/data/att{i}.wav', 1.2, sp)

            line = line[:-1]
            print(' | '.join([str(x) for x in line]))
        print('\n')
        print('att_size', curr_attention[0].shape)


def one_iteration(model, model_optim, loader, args, iteration, dataset_type, **kwargs):
    loader_dict, min_spec_len, iteration = safe_next_batch(args, loader, model, dataset_type, iteration)
    phoneme_input = loader_dict.get("x_phoneme")
    target_mel = loader_dict.get("y_specM_whole")
    spkr_id = loader_dict.get("idx_speaker")
    phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
    spec_lengths = loader_dict.get("len_spec")
    data_id = loader_dict.get("idx_data")

    with torch.no_grad():
        N = phoneme_input.size(0)
        model_out_dict = model.align(
            phoneme_input,
            phoneme_lengths,
            target_mel,
            spec_lengths.tolist(),
            debug=args.debug
            )
        attention = model_out_dict.get('attention')
        att_nll = model_out_dict.get('att_nll')
        att_key = model_out_dict.get('att_key')
        
    return spkr_id.item(), data_id[0], att_nll.item(), attention, target_mel, phoneme_input, att_key

def safe_next_batch(args, loader, model, dataset_type, iteration):
    loader_dict = loader.next_batch(dataset_type)
    subbatch_spec_lengths = loader_dict.get("subbatch_len_spec")
    min_spec_len = min(subbatch_spec_lengths)
    iteration += 1

    return loader_dict, min_spec_len, iteration


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
