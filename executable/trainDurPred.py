# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, sys
num_core = os.cpu_count()
if num_core == 20:
    os.putenv("OMP_NUM_THREADS", str(num_core // 4))
else:
    os.putenv("OMP_NUM_THREADS", str(num_core // 2))

import multiprocessing, subprocess, time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tts_text_util.get_vocab import get_sg_vocab
import voxa
from voxa.database.data_group import expand_data_group

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.util.default_args import get_default_training_args, get_backward_compatible_args
from tacotron.loader import DataLoader
from tacotron.util.train_common import verify_args, decay_learning_rate, set_default_GST, timeSince


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
    loss_list = []
    l1_loss_total = 0  # Reset every print_every
    prosody_loss_total = 0  # Reset every print_every
    context_loss_total = 0  # Reset every print_every
    spkr_recog_loss_total = 0  # Reset every print_every
    val_loss = 0
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

    # post_output_size must be the same with the decoder_output_size
    args.post_out_size = args.dec_out_size = voxa_config.mel_dim

    # reuse args in checkpoint to prevent incompatibility
    args.data = ckpt_args.data
    args.config = ckpt_args.config
    args.use_sg = ckpt_args.use_sg
    args.charvec_dim = ckpt_args.charvec_dim
    args.enc_hidden = ckpt_args.enc_hidden
    args.att_hidden = ckpt_args.att_hidden
    args.dec_hidden = ckpt_args.dec_hidden
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
    args.prosody_ignore_spkrs = ckpt_args.prosody_ignore_spkrs

    # set trunc_size to spec_limit (no TBPTT)
    args.trunc_size = args.spec_limit

    if args.gst == 0:
        from tacotron.model_finegrained import Tacotron as Tacotron
    elif args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron
        if args.n_head == 0:
            print('Using contents based attention')
        elif args.n_head > 0:
            print('Using Vaswani attention (Attention is all you need)')
    elif args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron

    # set dataset option
    if args.data is None:
        print('no dataset')
        return
    else:
        dataset_list = sorted(expand_data_group(args.data.split(','), voxa_config))
    args.data = ",".join(dataset_list)              # save explicitly.

    loader = DataLoader(args, dataset_list, voxa_config, speaker_list=speaker_list)
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
    model_optim = optim.Adam(model.duration_predictor.parameters(), args.learning_rate)
    state_dict = checkpoint['state_dict']

    # ignore the predictor in checkpoint
    for key in model.state_dict().keys():
        if key.startswith('duration_predictor'):
            state_dict[key] = model.state_dict()[key]
    keys_to_pop = []
    for key in state_dict:
        if not key in model.state_dict().keys():
            keys_to_pop.append(key)
    for key in keys_to_pop:
        state_dict.pop(key)

    model.load_state_dict(state_dict)
    
    # set speakers to ignore for prosody prediction
    if args.prosody_ignore_spkrs is not None:
        prosody_ignore_spkrs = sorted(expand_data_group(args.prosody_ignore_spkrs, voxa_config))
        args.prosody_ignore_spkrs = ",".join(prosody_ignore_spkrs)

        prosody_ignore_mask = [1 for x in range(args.num_id)]
        for spkr_id in prosody_ignore_spkrs:
            if loader.speaker_manager.get_compact_id(spkr_id) is None:
                raise RuntimeError('invalid speaker id in prosody_ignore_spkrs.')
            prosody_ignore_mask[loader.speaker_manager.get_compact_id(spkr_id)] = 0
        prosody_ignore_mask = torch.Tensor(prosody_ignore_mask)
    else:
        prosody_ignore_mask = None

    # set single/multi gpu usage
    if args.gpu:
        torch.cuda.manual_seed(0)
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.duration_predictor.parameters():
        param.requires_grad = True
    model = model.train()
    
    # print('set GST')
    # set_default_GST(loader, model, args)

    start = time.time()
    print('Start training... (1 epoch = %s iters)' % (iter_per_epoch))
    while iteration < max_iters + 1:
        loss_dict = one_iteration(model, model_optim, loader, args, iteration, loader_train_flag)

        iteration = loss_dict['iteration']
        l1_loss_total += loss_dict['loss']

        # compute validation loss at every epoch
        if args.no_validation == 0 and \
            loader.split_cursor[loader_train_flag] == 0 and loader.is_subbatch_end[loader_train_flag]:
            val_loss = 0
            model.reset_decoder_states()
            with torch.no_grad():
                while True:
                    val_loss_dict = one_iteration(model, model_optim, loader, args, iteration, 'valid')
                    val_loss += val_loss_dict['loss']
                    if loader.split_cursor['valid'] == 0 and loader.is_subbatch_end['valid']:
                        break
                val_loss = val_loss / loader.split_num_iters['valid']

        if iteration % args.print_every == 0:
            l1_loss_avg = l1_loss_total / args.print_every
            loss_list.append(l1_loss_avg)
            l1_loss_total = 0

            print('%s (%d) %.4f %.4f' % (
                timeSince(start, iteration / max_iters),
                iteration,
                l1_loss_avg,
                val_loss))

        if iteration % args.save_every == 0 or (iteration == max_iters):
            model.reset_decoder_states()
            epoch = iteration // iter_per_epoch
            if iteration >= max_iters:
                save_name = '%s/%s_last_dp.t7' % (args.save_dir, args.exp_no)
            else:
                save_name = '%s/%s_%d_dp.t7' % (args.save_dir, args.exp_no, epoch)
            state = {
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'loss_list': loss_list,
                'config': voxa_config
            }

            if 'cluster_list' in checkpoint.keys():
                state['cluster_list'] = checkpoint['cluster_list']
            if 'iteration' in checkpoint.keys():
                state['iteration'] = checkpoint['iteration']

            torch.save(state, save_name)
            print('model saved to', os.path.abspath(save_name))

            if iteration % args.veripe_every == 0:
                veripe_checkpoint =  os.path.join(ROOT_PATH, 'checkpoint/', save_name.split('/')[-1])
                cmd_copy_push = f'cp {os.path.abspath(save_name)} {os.path.join(ROOT_PATH, "checkpoint/")}'

                if args.gst == 0 or args.gst == 1 or args.gst == 2:
                    cmd_set = f'python executable/evaluation.py --target_type wav --server 1'
                    subprocess.call(['veripe', 'set', 'cmd', cmd_set])

                    cmd_push = f'veripe push --model-path {veripe_checkpoint} --exp-no {args.exp_no} --epoch {epoch}'
                    cmd_copy_push = cmd_copy_push + f'; {cmd_push}'
                    # if is_best:               # TODO: implement saving best model.
                    #     shutil.copyfile(save_name, '%s/%d_best.t7' % (args.save_dir, args.exp_no))

                subprocess.Popen(cmd_copy_push, shell=True, cwd=ROOT_PATH)


def one_iteration(model, model_optim, loader, args, iteration, dataset_type, **kwargs):
    loader_dict, min_spec_len, iteration = safe_next_batch(args, loader, model, dataset_type, iteration)
    phoneme_input = loader_dict.get("x_phoneme")
    target_mel_whole = loader_dict.get("y_specM_whole")
    spkr_id = loader_dict.get("idx_speaker")
    phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
    spec_lengths = loader_dict.get("len_spec")
    prosody_ignore_mask = kwargs.get('prosody_ignore_mask')

    if args.lr_decay == 1:
        for param_group in model_optim.param_groups:
            param_group['lr'] = decay_learning_rate(args.learning_rate, iteration)

    model_optim.zero_grad()

    with torch.no_grad():
        N = phoneme_input.size(0)
        text_lengths, spec_lengths, text_mask, spec_mask, whole_spec_mask = model.get_seq_mask(N, phoneme_lengths, spec_lengths.tolist(), spec_lengths.tolist(), phoneme_input.device)

        _, enc_output = model.encoder(
            phoneme_input,
            text_lengths,
            text_mask,
            debug=args.debug
        )

        # attention
        short_token_mask = torch.nn.functional.embedding(phoneme_input, model.short_token)
        attention_ref, att_loss, _, _ = model.attention(
            phoneme_input,
            target_mel_whole,
            text_lengths,
            spec_lengths,
            args.debug,
            enc_input=phoneme_input,
            short_token_mask=short_token_mask,
            text_mask=text_mask,
        )
        
        spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
        ref_out_dict = model.ref_encoder(target_mel_whole, whole_spec_mask, spkr_vec, debug=args.debug)  # N x 1 x style_dim
        gst_vec = ref_out_dict['gst']
        dec_input_from_enc = enc_output + gst_vec

        duration_gt = torch.sum(attention_ref, -1)
        text_mask = torch.arange(0, phoneme_input.size(1), device=phoneme_input.device).view(1, -1).expand(N, -1)
        text_mask = torch.lt(text_mask, text_lengths.view(-1, 1).expand(-1, phoneme_input.size(1)))                     # N x T_enc

        speed = text_lengths.type_as(spkr_vec) / spec_lengths.float()  # N

    duration_pred = model.duration_predictor(
            dec_input_from_enc.transpose(1, 2).detach(), 
            text_mask.unsqueeze(1),
            spkr_vec=spkr_vec.detach(),
            speed=speed,
            gst=gst_vec,
            text_lengths=text_lengths,
            spec_lengths=spec_lengths,
            debug=args.debug,
        )

    if prosody_ignore_mask is not None:
        prosody_ignore_mask = torch.index_select(prosody_ignore_mask, 0, spkr_id).type_as(duration_pred).view(N, 1, 1)
    else:
        prosody_ignore_mask = 1

    loss = torch.sum((duration_pred - duration_gt)**2 * prosody_ignore_mask) \
        / torch.sum(text_lengths * prosody_ignore_mask) * 0.01
    loss_primitive = loss.data.item()

    if dataset_type != 'valid':
        loss.backward()
        nn.utils.clip_grad_norm_(model.duration_predictor.parameters(), args.grad_clip)  # gradient clipping
        model_optim.step()

    return {
        'loss': loss_primitive,
        'iteration': iteration,
    }


def safe_next_batch(args, loader, model, dataset_type, iteration):
    min_spec_len = -1
    while min_spec_len <= args.r_factor * 2:
        if loader.is_subbatch_end[dataset_type]:
            model.reset_decoder_states(debug=args.debug)
        else:
            if model.is_reset:
                loader_dict = loader.next_batch(dataset_type)
                print(f'There is something wrong in [{dataset_type}] loader.')

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

