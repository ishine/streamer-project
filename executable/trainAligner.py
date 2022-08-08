# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, sys
num_core = os.cpu_count()
if num_core == 20:
    os.putenv("OMP_NUM_THREADS", str(num_core // 4))
else:
    os.putenv("OMP_NUM_THREADS", str(num_core // 2))

import multiprocessing, time

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
from tacotron.model_gst import Tacotron
from tacotron.util.train_common import verify_args, decay_learning_rate, timeSince


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
    args.spec_limit = args.trunc_size

    torch.manual_seed(0)
    np.random.seed(0)
    loss_list = []
    l1_loss_total = 0  # Reset every print_every
    train_nll_total = 0
    train_nll_avg = 0
    val_loss = 0
    val_nll = 0
    iteration = 0

    # load checkpoint file and set voxa_config
    speaker_list = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)

        # set parameters to freeze
        if 's' in args.freeze.split(','):
            # When do we freeze speaker? Decoder pretraining?
            speaker_list = sorted(checkpoint['args'].data.split(','))
            args.data = checkpoint['args'].data

        ckpt_voxa_config = voxa.VoxaConfig.load_from_config(checkpoint['config'], initialize=True)
        if args.config is None:
            voxa_config = ckpt_voxa_config
            voxa_config.refresh_env_variables()
        else:
            voxa_config = voxa.VoxaConfig(args.config)
            if not voxa_config.is_identical(ckpt_voxa_config):
                raise RuntimeError('voxa_config is incompatible with the checkpoint.')
    else:
        checkpoint = None
        if args.config is None:
            args.config = voxa.default_voxa_config_path
        voxa_config = voxa.VoxaConfig(args.config)

    if args.dec_out_type == 'mel':
        # post_output_size must be the same with the decoder_output_size
        args.post_out_size = args.dec_out_size = voxa_config.mel_dim
    else:
        args.post_out_size = int(voxa_config.n_fft // 2) + 1
        args.dec_out_size = voxa_config.mel_dim

    # set dataset option
    if args.data is None:
        print('no dataset')
        return
    else:
        dataset_list = sorted(expand_data_group(args.data.split(','), voxa_config))
    args.data = ",".join(dataset_list)              # save explicitly.

    # define loader
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
            args.print_every = max(1, int(max_epoch / 20) * iter_per_epoch)
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
    model_optim = optim.Adam(model.parameters(), args.learning_rate)

    # load model parameters 
    if args.init_from:
        # resume training
        if args.resume != 0:
            loss_list = checkpoint['loss_list']
            if 'iteration' in checkpoint:
                iteration =  checkpoint['iteration']
            else:
                iteration =  checkpoint['epoch'] * iter_per_epoch + 1
        print(f"load checkpoint {args.init_from} (epoch {checkpoint['epoch']})")

        state_dict = checkpoint['state_dict']
    else:
        state_dict = model.state_dict()
    model.load_state_dict(state_dict)

    # set single/multi gpu usage
    if args.gpu:
        torch.cuda.manual_seed(0)
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model = model.cuda()
    model = model.train()

    start = time.time()
    print('Start training... (1 epoch = %s iters)' % (iter_per_epoch))
    print('time_elapsed | eta | iteration | train_loss | att_nll_train | valid_loss | att_nll_valid')
    while iteration < max_iters + 1:
        loss_dict = one_iteration(model, model_optim, loader, args, iteration, loader_train_flag, train_nll_avg=train_nll_avg)
        iteration = loss_dict['iteration']
        l1_loss_total += loss_dict['loss']
        train_nll_total += loss_dict['train_nll']

        # compute validation loss at every epoch
        if args.no_validation == 0 and \
            loader.split_cursor[loader_train_flag] == 0 and loader.is_subbatch_end[loader_train_flag]:
            val_loss = 0
            val_nll = 0
            with torch.no_grad():
                while True:
                    val_loss_dict = one_iteration(model, model_optim, loader, args, iteration, 'valid', train_nll_avg=train_nll_avg)
                    val_loss += val_loss_dict['loss']
                    val_nll += val_loss_dict['train_nll']
                    if loader.split_cursor['valid'] == 0 and loader.is_subbatch_end['valid']:
                        break
                val_loss = val_loss / loader.split_num_iters['valid']
                val_nll = val_nll / loader.split_num_iters['valid']

        if iteration % args.print_every == 0:
            l1_loss_avg = l1_loss_total / args.print_every
            loss_list.append(l1_loss_avg)
            l1_loss_total = 0
            train_nll_avg = 0

            if train_nll_total != 0:
                train_nll_avg = train_nll_total / args.print_every
                train_nll_total = 0

            print('%s (%d) %.4f %.4f | %.4f %.4f' % (
                timeSince(start, iteration / max_iters),
                iteration,
                l1_loss_avg,
                train_nll_avg,
                val_loss,
                val_nll))

        if iteration % args.save_every == 0 or (iteration == max_iters):
            epoch = iteration // iter_per_epoch
            if iteration >= max_iters:
                save_name = '%s/%s_last.t7' % (args.save_dir, args.exp_no)
            else:
                save_name = '%s/%s_%d.t7' % (args.save_dir, args.exp_no, epoch)
            state = {
                'iteration': iteration,
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'loss_list': loss_list,
                'config': voxa_config
            }

            torch.save(state, save_name)
            print('model saved to', os.path.abspath(save_name))


def one_iteration(model, model_optim, loader, args, iteration, dataset_type, **kwargs):
    loader_dict, iteration = safe_next_batch(loader, dataset_type, iteration)
    phoneme_input = loader_dict.get("x_phoneme")
    target_mel = loader_dict.get("y_specM")
    phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
    spec_lengths = loader_dict.get("len_spec")

    if args.lr_decay == 1:
        for param_group in model_optim.param_groups:
            param_group['lr'] = decay_learning_rate(args.learning_rate, iteration)

    model_optim.zero_grad()

    # train aux_aligner
    model_out_dict = model.align(
        phoneme_input,
        phoneme_lengths,
        target_mel,
        spec_lengths.tolist(),
        debug=args.debug
        )
    att_nll = model_out_dict.get('att_nll')
    att_loss = model_out_dict.get('att_loss')
    loss_primitive = att_loss.data.item()
    train_nll_primitive = att_nll.data.item()

    if dataset_type != 'valid':
        att_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # gradient clipping
        model_optim.step()

    return {
        'loss': loss_primitive,
        'train_nll': train_nll_primitive,
        'iteration': iteration,
    }


def safe_next_batch(loader, dataset_type, iteration):
    loader_dict = loader.next_batch(dataset_type)
    iteration += 1
    return loader_dict, iteration


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
