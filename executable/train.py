# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import multiprocessing, subprocess, time
import os, sys
num_core = os.cpu_count()
if num_core == 20:
    os.putenv("OMP_NUM_THREADS", str(num_core // 4 // 2))
else:
    os.putenv("OMP_NUM_THREADS", str(num_core // 2 // 2))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tts_text_util.get_vocab import get_sg_vocab
import voxa
from voxa.database.data_group import expand_data_group

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.loader import DataLoader
from tacotron.util.default_args import get_default_training_args, get_backward_compatible_args
from tacotron.util.train_common import verify_args, adapt_from_ckpt, get_adapted_phoneme_embedding_matrix, decay_learning_rate, set_default_GST, timeSince


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
    assert args.trunc_size % args.r_factor == 0
    assert args.veripe_every % args.save_every == 0

    if args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron
    elif args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron
            
    torch.manual_seed(0)
    np.random.seed(0)
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

    # post_output_size must be the same with the decoder_output_size
    args.post_out_size = args.dec_out_size = voxa_config.mel_dim

    # set dataset option
    if args.data is None:
        print('no dataset')
        return
    else:
        dataset_list = sorted(expand_data_group(args.data, voxa_config))
    args.data = ",".join(dataset_list)              # save explicitly.

    # set misc options
    os.makedirs(args.save_dir, exist_ok=True)               # make save directory if not exists.
    vocab, idx_to_vocab = get_sg_vocab()
    args.vocab_size = len(vocab)

    # set loader
    loader = DataLoader(args, dataset_list, voxa_config, speaker_list=speaker_list)
    args.num_id = loader.speaker_manager.get_num_speakers()
    if hasattr(loader, 'language_manager'):
        args.num_lang_id = loader.language_manager.get_num_languages()
    else:
        args.num_lang_id = 0
    loader_train_flag = 'train'

    # prepare for logging
    loss_keys = [
        'mel_train',
        'misc',
        'duration_pred',
        'att_nll_train',
        'prosody_loss',
        'mel_valid',
        'dur_pred_valid',
        'spkr_loss',
        'lang_loss',
        'adv_lang_loss',
    ]
    loss_list = []
    loss_list.append(' '.join(loss_keys))
    curr_epoch_total = dict.fromkeys(loss_keys, 0)

    # set logging period
    iteration = 0
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

    # initialize model
    model = Tacotron(args, vocab=vocab, idx_to_vocab=idx_to_vocab)
    model_optim = optim.Adam(model.parameters(), args.learning_rate)
    model_optim_mi = None
    
    # Write each speakers' max pitch value
    for spkr_id in loader.speaker_manager.get_all_compact_id():
        model.prosody_stats.put_max_pitch(spkr_id, loader.pitch_max[spkr_id])

    # TODO: modularize this part. [START]
    # load model parameters 
    if args.init_from:
        # resume training
        if args.resume != 0:
            # model_optim.load_state_dict(checkpoint['optimizer'])
            loss_list = checkpoint['loss_list']
            if 'iteration' in checkpoint:
                iteration =  checkpoint['iteration']
            else:
                iteration =  checkpoint['epoch'] * iter_per_epoch
        print(f"load checkpoint {args.init_from} (epoch {checkpoint['epoch']})")

        state_dict = checkpoint['state_dict']
        ckpt_args = checkpoint['args']
        if args.fluency == 1:
            lang_dict = checkpoint['lang_dict']        
            # adapt language embedding matrix from checkpoint
            old_language_list = list(lang_dict.keys())
            lang_embed_list = [
                'lang_embed.weight',
            ]
            for key in state_dict.keys():
                if key in lang_embed_list:
                    old_embedding_matrix = state_dict[key]
                    new_embedding_matrix = loader.speaker_manager.get_adapted_speaker_embedding_matrix(old_language_list, old_embedding_matrix)
                    state_dict[key] = new_embedding_matrix

        # adapt speaker embedding matrix from checkpoint
        old_data_list = ckpt_args.data.split(',')
        spkr_embed_list = [
            'spkr_embed.weight',
        ]
        for key in state_dict.keys():
            if key in spkr_embed_list:
                old_embedding_matrix = state_dict[key]
                new_embedding_matrix = loader.speaker_manager.get_adapted_speaker_embedding_matrix(old_data_list, old_embedding_matrix)
                state_dict[key] = new_embedding_matrix

        # adapt phoneme embedding matrix from checkpoint
        phoneme_embed_list = [
            'encoder.embedding.weight',
            'attention.embedding.weight',
            'attention.ctc_proj.weight',
            'attention.ctc_proj.bias',
            'short_token',
        ]
        for key in state_dict.keys():
            if key in phoneme_embed_list:
                old_embedding_matrix = state_dict[key]
                new_embedding_matrix = get_adapted_phoneme_embedding_matrix(len(vocab), old_embedding_matrix, key)
                state_dict[key] = new_embedding_matrix


        # adapt prosody statistics
        if 'prosody_stats.means' in state_dict.keys():
            state_dict['prosody_stats.means'] = adapt_from_ckpt(
                model.prosody_stats.means.data, 
                state_dict['prosody_stats.means']
            )
            state_dict['prosody_stats.question'] = adapt_from_ckpt(
                model.prosody_stats.question.data, 
                state_dict['prosody_stats.question']
            )
        if 'prosody_stats.speed' in state_dict.keys():
            state_dict['prosody_stats.speed'] = adapt_from_ckpt(
                model.prosody_stats.speed.data, 
                state_dict['prosody_stats.speed']
            )
        if 'prosody_stats.max_pitch' in state_dict.keys():
            state_dict['prosody_stats.max_pitch'] = adapt_from_ckpt(
                model.prosody_stats.max_pitch.data, 
                state_dict['prosody_stats.max_pitch']
            )

        if args.gst == 0 or args.gst == 2:
            if 'prosody_stats_predictor.means' in state_dict.keys():
                state_dict['prosody_stats_predictor.means'] = adapt_from_ckpt(
                    model.prosody_stats_predictor.means.data, 
                    state_dict['prosody_stats_predictor.means']
                )
                state_dict['prosody_stats_predictor.question'] = adapt_from_ckpt(
                    model.prosody_stats_predictor.question.data, 
                    state_dict['prosody_stats_predictor.question']
                )
            if 'prosody_stats_predictor.speed' in state_dict.keys():
                state_dict['prosody_stats_predictor.speed'] = adapt_from_ckpt(
                    model.prosody_stats_predictor.speed.data, 
                    state_dict['prosody_stats_predictor.speed']
                )
    else:
        state_dict = model.state_dict()

    # load a saved aligner
    if args.aligner_from:
        tmp_checkpoint = torch.load(args.aligner_from, map_location=lambda storage, loc: storage)
        ckpt_state_dict = tmp_checkpoint['state_dict']
        old_data_list = tmp_checkpoint['args'].data.split(',')
        spkr_embed_list = [
            'spkr_embed.weight',
        ]
        for key in ckpt_state_dict.keys():
            # if key.startswith('attention_aux'):
            if key.startswith('attention') or key.startswith('encoder') or key.startswith('ref_encoder'):
                if key in spkr_embed_list:
                    old_embedding_matrix = ckpt_state_dict[key]
                    new_embedding_matrix = loader.speaker_manager.get_adapted_speaker_embedding_matrix(old_data_list, old_embedding_matrix)
                    state_dict[key] = new_embedding_matrix
                else:
                    state_dict[key] = ckpt_state_dict[key]

    # # ignore the duration predictor in checkpoint
    # for key in model.state_dict().keys():
    #     if key.startswith('duration_predictor'):
    #         state_dict[key] = model.state_dict()[key]
    # keys_to_pop = []
    # for key in state_dict:
    #     if not key in model.state_dict().keys():
    #         keys_to_pop.append(key)
    # for key in keys_to_pop:
    #     state_dict.pop(key)

    model.load_state_dict(state_dict)
    # TODO: modularize this part. [END]
    
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
        if prosody_ignore_mask is not None:
            prosody_ignore_mask = prosody_ignore_mask.cuda()

    # freeze models if needed
    minimum_iter_to_save = 0
    if args.freeze != '':
        freeze_list = args.freeze.split(',')
        model.freeze_params(freeze_list)

    # pre-allocate memory
    if args.mem_prealloc == 1:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
        model = model.train()
        for _ in range(args.spec_limit//args.trunc_size):
            one_iteration(
                model,
                model_optim,
                loader,
                args,
                iteration,
                loader_train_flag,
                prosody_ignore_mask=prosody_ignore_mask,
                model_optim_mi=model_optim_mi,
                is_prealloc=True,
            )
        model.reset_decoder_states(debug=args.debug)

    # start training
    model = model.train()
    start = time.time()
    print('Start training... (1 epoch = %s iters)' % (iter_per_epoch))
    print(f'time_elapsed | eta | iteration | {" | ".join(loss_keys)}')
    while iteration < max_iters + 1:
        loss_dict = one_iteration(
            model,
            model_optim,
            loader,
            args,
            iteration,
            loader_train_flag,
            prosody_ignore_mask=prosody_ignore_mask,
            model_optim_mi=model_optim_mi
        )
        iteration = loss_dict['iteration']
        curr_epoch_total['mel_train'] += loss_dict['l1_loss']
        curr_epoch_total['misc'] += loss_dict['context_loss']
        curr_epoch_total['duration_pred'] += loss_dict['durpred_loss']
        curr_epoch_total['att_nll_train'] += loss_dict['att_nll']
        curr_epoch_total['prosody_loss'] += loss_dict['prosody_loss']
        curr_epoch_total['spkr_loss'] += loss_dict['spkr_loss']
        curr_epoch_total['lang_loss'] += loss_dict['lang_loss']
        curr_epoch_total['adv_lang_loss'] += loss_dict['adv_lang_loss']

        # validation
        if args.no_validation == 0 \
            and loader.split_cursor[loader_train_flag] == 0 \
            and loader.is_subbatch_end[loader_train_flag]:
            curr_epoch_total['mel_valid'] = 0
            curr_epoch_total['dur_pred_valid'] = 0
            model.reset_decoder_states()
            with torch.no_grad():
                while True:
                    val_loss_dict = one_iteration(
                        model,
                        model_optim,
                        loader,
                        args,
                        iteration,
                        'valid',
                        model_optim_mi=model_optim_mi
                    )
                    curr_epoch_total['mel_valid'] += val_loss_dict['l1_loss']
                    curr_epoch_total['dur_pred_valid'] += val_loss_dict['durpred_loss']
                    if loader.split_cursor['valid'] == 0 \
                        and loader.is_subbatch_end['valid']:
                        break
                curr_epoch_total['mel_valid'] = curr_epoch_total['mel_valid'] / loader.split_num_iters['valid']
                curr_epoch_total['dur_pred_valid'] = curr_epoch_total['dur_pred_valid'] / loader.split_num_iters['valid']

        # log numbers
        if iteration % args.print_every == 0:
            for key in curr_epoch_total.keys():
                if not key.endswith('valid'):
                    curr_epoch_total[key] /= args.print_every

            curr_losses = ['%.4f' % curr_epoch_total[x] for x in loss_keys]
            curr_losses = f"({iteration}) {' '.join(curr_losses)}"
            loss_list.append(curr_losses)
            elapsed_time = str(timeSince(start, iteration / max_iters))
            print(f'{elapsed_time} {curr_losses}')

            att_nll_train_threshold = 2.2
            if curr_epoch_total['att_nll_train'] < att_nll_train_threshold:
                model.train_durpred = True

            for key in curr_epoch_total.keys():
                if not key.endswith('valid'):
                    curr_epoch_total[key] = 0

        # log file
        if (iteration % args.save_every == 0 and iteration > minimum_iter_to_save) \
            or (iteration == max_iters):
            # compute default global style
            if (iteration % args.veripe_every == 0) \
                or (iteration == max_iters) \
                or (iteration == args.save_every):
                if args.gst == 1:
                    print('set GST')
                    set_default_GST(loader, model, args)
                else:
                    pass

            # save spectrogram generator
            model.reset_decoder_states()
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
                'config': voxa_config,
                'lang_dict': loader.lang_dict
            }
            torch.save(state, save_name)
            print(f'model saved: {os.path.abspath(save_name)}')

            # veripe push
            if iteration % args.veripe_every == 0 and iteration > minimum_iter_to_save:
                veripe_checkpoint =  os.path.join(ROOT_PATH, 'checkpoint/', save_name.split('/')[-1])
                cmd_copy_push = f'cp {os.path.abspath(save_name)} {os.path.join(ROOT_PATH, "checkpoint/")}'
                cmd_set = f'python executable/evaluation.py --target_type wav --server 1 --eval_set {args.eval_set}'
                subprocess.call(['veripe', 'set', 'cmd', cmd_set])
                cmd_push = f'veripe push --model-path {veripe_checkpoint} --exp-no {args.exp_no} --epoch {epoch}'
                cmd_copy_push = cmd_copy_push + f'; {cmd_push}'
                subprocess.Popen(cmd_copy_push, shell=True, cwd=ROOT_PATH)


def one_iteration(model, model_optim, loader, args, iteration, dataset_type, is_prealloc=False, **kwargs):
    if is_prealloc:
        loader_dict, min_spec_len, iteration = get_dummy_max_batch(args, model, iteration)
    else:
        loader_dict, min_spec_len, iteration = safe_next_batch(args, loader, model, dataset_type, iteration)
    phoneme_input = loader_dict.get("x_phoneme")
    target_mel = loader_dict.get("y_specM")
    target_mel_whole = loader_dict.get("y_specM_whole")
    target_pitch = loader_dict.get("y_pitch")
    target_pitch_whole = loader_dict.get("y_pitch_whole")
    spkr_id = loader_dict.get("idx_speaker")
    subbatch_spec_lengths = loader_dict.get("subbatch_len_spec")
    phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
    spec_lengths = loader_dict.get("len_spec")
    len_mask = loader_dict.get("len_spec_mask")
    prosody_ignore_mask = kwargs.get('prosody_ignore_mask')
    batch_size = phoneme_input.size(0)
    lang_id = loader_dict.get('lang_id')

    # learning rate decay
    if args.lr_decay == 1:
        for param_group in model_optim.param_groups:
            param_group['lr'] = decay_learning_rate(args.learning_rate, iteration)

    # Train Spectrogram generator
    model.mask_decoder_states(len_mask, debug=args.debug)
    model_optim.zero_grad(set_to_none=True)
    model_out_dict = model(
        phoneme_input,
        target_mel,
        spkr_id,
        subbatch_spec_lengths,
        phoneme_lengths,
        target_mel_whole=target_mel_whole,
        whole_spec_len=spec_lengths,
        debug=args.debug,
        exp_no=args.exp_no,
        prosody_ignore_mask=prosody_ignore_mask,
        target_pitch=target_pitch,
        target_pitch_whole=target_pitch_whole,
        lang_id=lang_id
    )
    pred_mel = model_out_dict.get('output_dec')
    pred_lin = model_out_dict.get('output_post')

    # consider spectral flatness
    if args.spectral_flatness == 1:
        geo_mean_mel = torch.exp(torch.mean(target_mel.data[:, :min_spec_len], dim=-1))
        ari_mean_mel = torch.mean(torch.exp(target_mel.data[:, :min_spec_len]), dim=-1)
        flatness_weight = (1 - geo_mean_mel / ari_mean_mel) * 200
    else:
        flatness_weight = 1

    # spectrogram l1 loss
    mel_normalizing_factor = min_spec_len * batch_size * args.dec_out_size
    lin_normalizing_factor = min_spec_len * batch_size * args.post_out_size
    loss_dec = torch.sum(torch.sum(
        F.l1_loss(
            pred_mel[:, :min_spec_len],
            target_mel[:, :min_spec_len],
            reduction='none'
        ),
        dim=-1
    ) * flatness_weight)
    loss_post = torch.sum(torch.sum(
        F.l1_loss(
            pred_lin[:, :min_spec_len],
            target_mel[:, :min_spec_len],
            reduction='none'
        ), 
        dim=-1
    ) * flatness_weight)
    loss = loss_dec.div(mel_normalizing_factor) + loss_post.div(lin_normalizing_factor)

    if args.aug_teacher_forcing == 1:
        pred_mel2 = model_out_dict.get('output_dec2')
        pred_lin2 = model_out_dict.get('output_post2')
        if pred_mel2 is not None and pred_lin2 is not None:
            loss_dec2 = torch.sum(torch.sum(
                F.l1_loss(
                    pred_mel2[:, :min_spec_len],
                    target_mel[:, :min_spec_len],
                    reduction='none'
                ),
                dim=-1
            ) * flatness_weight)
            loss_post2 = torch.sum(torch.sum(
                F.l1_loss(
                    pred_lin2[:, :min_spec_len],
                    target_mel[:, :min_spec_len],
                    reduction='none'
                ),
                dim=-1
            ) * flatness_weight)
            loss += loss_dec2.div(mel_normalizing_factor) + loss_post2.div(lin_normalizing_factor)
    l1_loss_primitive = loss.data.item()

    # misc loss
    prenet_loss = model_out_dict.get('prenet_loss', 0)
    spkr_adv_loss = model_out_dict.get('spkr_adv_loss', 0)
    stop_loss = model_out_dict.get('stop_loss', 0)
    ctx_loss = model_out_dict.get('ctx_loss', 0)
    ctx_loss = prenet_loss + spkr_adv_loss + stop_loss + ctx_loss
    context_loss_primitive = ctx_loss.data.item()
    loss += ctx_loss

    # attention loss
    att_loss = model_out_dict.get('att_loss', 0)
    att_loss_primitive = att_loss.data.item()
    loss += att_loss

    # duration predictor loss
    durpred_loss = model_out_dict.get('durpred_loss', 0)
    durpred_loss_primitive = durpred_loss.data.item()
    loss += durpred_loss

    # prosody loss
    tside_prosody_loss = model_out_dict.get('tside_prosody_loss', 0)
    sside_prosody_loss = model_out_dict.get('sside_prosody_loss', 0)
    prosody_loss = tside_prosody_loss + sside_prosody_loss
    if type(prosody_loss) is int:
        prosody_loss_primitive = prosody_loss
    else:
        prosody_loss_primitive = prosody_loss.data.item()
    loss += prosody_loss
    
    if args.fluency == 1:
        spkr_vec=model_out_dict.get('spkr_vec')
        lang_vec=model_out_dict.get('lang_vec')
        spkr_loss = nn.functional.cross_entropy(model.spkr_classifier(spkr_vec.squeeze(1)), spkr_id)
        lang_loss = nn.functional.cross_entropy(model.lang_classifier(lang_vec.squeeze(1)), lang_id)
        adv_lang_loss = nn.functional.cross_entropy(model.adv_lang_classifier(spkr_vec.squeeze(1)), lang_id)
        spkr_loss_primitive = spkr_loss.data.item()
        lang_loss_primitive = lang_loss.data.item()
        adv_lang_loss_primitive = adv_lang_loss.data.item()
        loss += (spkr_loss + lang_loss + adv_lang_loss)
    else:
        spkr_loss_primitive = 0
        lang_loss_primitive = 0
        adv_lang_loss_primitive = 0


    if dataset_type != 'valid':
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # gradient clipping
        if not is_prealloc:
            model_optim.step()
    return {
        'att_nll': att_loss_primitive,
        'l1_loss': l1_loss_primitive,
        'prosody_loss': prosody_loss_primitive,
        'context_loss': context_loss_primitive,
        'durpred_loss': durpred_loss_primitive,
        'iteration': iteration,
        'spkr_loss': spkr_loss_primitive,
        'lang_loss': lang_loss_primitive,
        'adv_lang_loss' : adv_lang_loss_primitive,
    }


def safe_next_batch(args, loader, model, dataset_type, iteration):
    if model is not None:
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


def get_dummy_max_batch(args, model, iteration):
    device=next(model.parameters()).device
    target_mel_whole = torch.rand(args.batch_size, args.spec_limit, args.dec_out_size, device=device)
    target_pitch_whole = torch.randint(0, 880, [args.batch_size, args.spec_limit], device=device).float()
    spkr_id = torch.randint(0, args.num_id, [args.batch_size], device=device)
    lang_id = torch.randint(0, args.num_lang_id, [args.batch_size], device=device)
    loader_dict = {
        'x_phoneme': torch.randint(0, args.vocab_size, [args.batch_size, args.text_limit], device=device),
        'y_specM': target_mel_whole[:, :args.trunc_size],
        'y_specM_whole': target_mel_whole,
        'y_pitch': target_pitch_whole[:, :args.trunc_size],
        'y_pitch_whole': target_pitch_whole,
        'idx_speaker': spkr_id,
        'subbatch_len_spec': [args.trunc_size for _ in range(args.batch_size)],
        'subbatch_len_phoneme': [args.text_limit for _ in range(args.batch_size)],
        'len_spec': torch.tensor([args.spec_limit for _ in range(args.batch_size)], device=device),
        'len_spec_mask': torch.arange(0, args.batch_size, 1, device=device),
        'lang_id': lang_id
        
    }
    min_spec_len = args.trunc_size
    iteration += 1

    # simulate set_default_GST
    with torch.no_grad():
        new_spec_limit = 1000
        target_mel_whole = torch.rand(args.batch_size, new_spec_limit, args.dec_out_size, device=device)
        spec_lengths = torch.tensor([new_spec_limit for _ in range(args.batch_size)], device=device)
        spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S            
        whole_spec_mask = torch.arange(0, spec_lengths.max(), device=spec_lengths.device).view(1, -1).expand(spec_lengths.size(0), -1)
        whole_spec_mask = torch.lt(whole_spec_mask, spec_lengths.view(-1, 1).expand(-1, spec_lengths.max()))             # N x T_dec
        model.ref_encoder(
            target_mel_whole,
            whole_spec_mask,
            spkr_vec,
            debug=args.debug
        )
    return loader_dict, min_spec_len, iteration


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
