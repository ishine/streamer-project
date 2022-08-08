# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from functools import partial
import os.path

import numpy as np
import torch, pickle
from torch.multiprocessing import Process, Queue, Pool, set_start_method
from voxa.speakermgr.speaker_manager import SpeakerManager
import voxa
from packaging import version
from tacotron.util.gen_common import pad_text, normalize_pitch
from tts_text_util.get_vocab import phoneme2lang, get_sg_vocab, others, symbols
from tts_text_util.process_text_input import supported_languages

set_start_method('spawn', force=True)

class DataLoader():
    def __init__(self, args, dataset_list, voxa_config, speaker_list=None, sort=True, ssd=True):
        self.batch_size = args.batch_size
        self.r_factor = args.r_factor
        self.dec_out_size = args.dec_out_size
        self.post_out_size = args.post_out_size
        self.pretraining = args.pretraining
        self.shuffle_data = True if args.shuffle_data == 1 else False
        self.iter_per_epoch = None
        self.curr_split = None
        self.trunc_size = {'train': args.trunc_size, 'valid': args.trunc_size, 'whole': -1}
        self.is_subbatch_end = {'train': True, 'valid': True, 'whole': True}
        self.load_list = {'train': None, 'valid': None, 'whole': None}
        self.split_num_iters = {'train': None, 'valid': None, 'whole': None}
        self.split_sizes = {'train': None, 'valid': None, 'whole': None}
        self.split_cursor = {'train': None, 'valid': None, 'whole': None}
        self.common_multiple = {'train': None, 'valid': None, 'whole': None}
        self.path_dict = {'phoneme': {}, 'linear': {}, 'mel': {}, 'pitch': {}}
        self.len_mask_mapper = {}
        self.BOS_idx = 0
        self.EOS_idx = 1
        self.debug = args.debug
        offset_phone = 'offset_phone'
        read_n_phone = 'read_n_phone'
        
        # TODO: load lang dict from tts_text_util
        # TODO: delete below if voxa is updated
        self.lang_dict = {}
        if version.parse(voxa.__version__) > version.parse("0.3.68"):
            from voxa.langmgr.language_manager import LanguageManager
            self.language_manager = LanguageManager(supported_languages)
            self.lang_dict = self.language_manager.original_to_compact


        if args.no_validation == 0:
            train_ratio = 0.9
        else:
            train_ratio = 1
        valid_ratio = 1 - train_ratio

        if ssd:
            binary_root = voxa_config.ssd_bin_root
        else:
            binary_root = voxa_config.hdd_bin_root

        self.dataset_list = dataset_list
        if speaker_list is None:
            speaker_list = dataset_list
        else:
            invalid_speakers = set(dataset_list) - set(speaker_list)
            if len(invalid_speakers) > 0:
                invalid_speakers_str = ', '.join(list(invalid_speakers))
                print(f"{len(invalid_speakers)} invalid speakers: {invalid_speakers_str}")
                raise RuntimeError("Invalid dataset_list and speaker_list.")

        if hasattr(voxa_config, 'pitch_binary'):
            self.use_pitch = True
            self.pitch_means = [0 for _ in range(len(speaker_list))]
            self.pitch_stdevs = [0 for _ in range(len(speaker_list))]
            self.pitch_max = [0 for _ in range(len(speaker_list))]
            self.pitch_counts = [0 for _ in range(len(speaker_list))]
        else:
            self.use_pitch = False

        self.use_gpu = not (len(args.gpu) == 0)

        self.process = {'train': None, 'valid': None, 'whole': None}
        self.queue = {'train':  Queue(maxsize=args.load_queue_size),
                      'valid':  Queue(maxsize=args.load_queue_size),
                      'whole': Queue(maxsize=args.load_queue_size)}
        self.n_workers = args.n_workers

        dir_bin = {}
        with open(voxa_config.dataset_list_file, 'r') as rFile:
            for line in rFile.readlines():
                dataset, _, curr_dir_bin = line.strip().split('|')
                dir_bin[dataset] = os.path.join(binary_root, curr_dir_bin, voxa_config.alias)

        if os.path.isfile(args.filter_data_file):
            self.filter_data_list = torch.load(args.filter_data_file)
        else:
            self.filter_data_list = {}

        self.speaker_manager = SpeakerManager(speaker_list)
        self.speaker_occurrence = [0 for _ in range(self.speaker_manager.get_num_speakers())]
        for bin_file_idx, dataset in enumerate(self.dataset_list):
            cpt_spkr_id = self.speaker_manager.get_compact_id(dataset)
            if cpt_spkr_id is None:
                continue  # skip speakers whom we don't consider

            self.path_dict['phoneme'][bin_file_idx] = os.path.join(dir_bin[dataset], voxa_config.phoneme_sg_binary)
            self.path_dict['linear'][bin_file_idx] = os.path.join(dir_bin[dataset], voxa_config.linear_spec_binary)
            self.path_dict['mel'][bin_file_idx] = os.path.join(dir_bin[dataset], voxa_config.mel_spec_binary)
            if hasattr(voxa_config, 'pitch_binary'):
                self.path_dict['pitch'][bin_file_idx] = os.path.join(dir_bin[dataset], voxa_config.pitch_binary)
            load_list_file_path = os.path.join(dir_bin[dataset], voxa_config.load_dict_file)
            load_list_tmp = torch.load(load_list_file_path)              # N_data x 10 (or 9 if single speaker dataset)

            max_text_len = -1
            max_spec_len = -1
            filtered_load_list_tmp = []
            filtered_load_list_skipped = []
            for line in load_list_tmp['data']:
                data_idx = line['index']

                # To check etri_F's attention.
                # if data_idx != 238:
                # if data_idx != 3862:
                #     continue

                if dataset in self.filter_data_list and data_idx in self.filter_data_list[dataset]:
                    # filter out bad examples (based on pre-computed loss)
                    continue

                new_line = [
                    line['index'],
                    line['len_spec'],
                    line['len_text'],
                    line['offset_mel'],
                    line['offset_lin'],
                    line[offset_phone],
                    line['read_n_mel'],
                    line['read_n_lin'],
                    line[read_n_phone],
                ]
                if hasattr(voxa_config, 'pitch_binary'):
                    new_line.append(line['offset_pitch'])
                    new_line.append(line['read_n_pitch'])

                    pitch = load_binary(
                        self.path_dict['pitch'][bin_file_idx],
                        line['offset_pitch'],
                        line['read_n_pitch']
                    )
                    self.pitch_means[cpt_spkr_id] += np.sum(pitch)
                    self.pitch_stdevs[cpt_spkr_id] += np.sum(pitch ** 2)
                    self.pitch_max[cpt_spkr_id] = max(normalize_pitch(pitch.max()), self.pitch_max[cpt_spkr_id])
                    self.pitch_counts[cpt_spkr_id] += np.count_nonzero(pitch)

                new_line.append(cpt_spkr_id)
                new_line.append(bin_file_idx)

                self.trunc_size['whole'] = max(self.trunc_size['whole'], line['len_spec'])
                if line['len_spec'] > args.spec_limit or line['len_text'] > args.text_limit:
                    # # Uncomment below to use it when computing prosody.
                    # filtered_load_list_skipped.append(new_line)
                
                    # filter out whose spec/text length exceeds limit
                    pass
                else:
                    # add to load list
                    self.speaker_occurrence[cpt_spkr_id] += 1
                    filtered_load_list_tmp.append(new_line)
                    max_spec_len = max(max_spec_len, line['len_spec'])
                    max_text_len = max(max_text_len, line['len_text'])

            if hasattr(voxa_config, 'pitch_binary'):
                self.pitch_means[cpt_spkr_id] /= self.pitch_counts[cpt_spkr_id]
                self.pitch_stdevs[cpt_spkr_id] /= self.pitch_counts[cpt_spkr_id]
                self.pitch_stdevs[cpt_spkr_id] -= self.pitch_means[cpt_spkr_id] ** 2
                self.pitch_stdevs[cpt_spkr_id] = self.pitch_stdevs[cpt_spkr_id] ** 0.5

            filtered_load_list_tmp = torch.DoubleTensor(filtered_load_list_tmp)

            if len(filtered_load_list_skipped) > 0:
                filtered_load_list_skipped = torch.DoubleTensor(filtered_load_list_skipped)

            valid_set_length = int(len(filtered_load_list_tmp) * valid_ratio)
            if valid_set_length < 1 and args.no_validation == 0:
                print(f'Skipping {dataset}: not enough data to generate batches.')
            else:
                valid_set = filtered_load_list_tmp[:valid_set_length]
                if self.load_list['valid'] is None:
                    self.load_list['valid'] = valid_set
                else:
                    self.load_list['valid'] = torch.cat([self.load_list['valid'], valid_set], dim=0)

                train_set = filtered_load_list_tmp[valid_set_length:]
                if self.load_list['train'] is None:
                    self.load_list['train'] = train_set
                else:
                    self.load_list['train'] = torch.cat([self.load_list['train'], train_set], dim=0)

            if self.load_list['whole'] is None:
                self.load_list['whole'] = filtered_load_list_tmp
            else:
                self.load_list['whole'] = torch.cat([self.load_list['whole'], filtered_load_list_tmp], dim=0)
            if len(filtered_load_list_skipped) > 0:
                self.load_list['whole'] = torch.cat([self.load_list['whole'], filtered_load_list_skipped], dim=0)

            print(f'{dataset} | max text/spec length: {max_text_len} / {max_spec_len}')

        # col0: idx / col1: spec_length / col2: text_length
        # col3: offset_M / col4: offset_L / col5: offset_P
        # col6: len_binary_M / col7: len_binary_L / col8: len_binary_P
        # col9: speaker_id / col10: dataset_idx
        for split in self.load_list.keys():
            load_list = self.load_list[split]
            effective_batch_size = min(self.batch_size, len(load_list))

            if effective_batch_size == 0:
                self.split_sizes[split] = 0
                self.split_cursor[split] = 0
                self.split_num_iters[split] = 0
                continue

            # shuffle by element
            if self.shuffle_data:
                load_list = load_list[torch.randperm(len(load_list))]

            if sort:
                # sort by spec length
                spec_len_list = load_list[:, 1].clone()
                phoneme_len_list = load_list[:, 2].clone()
                sort_length, sort_idx = spec_len_list.sort(0, descending=True)
                phoneme_len_list = torch.gather(phoneme_len_list, 0, sort_idx)
                sort_idx = sort_idx.view(-1, 1).expand_as(load_list)
                load_list = torch.gather(load_list, 0, sort_idx)

            # drop residual data
            end_idx = len(load_list) - (len(load_list) % effective_batch_size)
            load_list = load_list[:end_idx]

            # print statistics
            num_total_sample = load_list.size(0)
            total_spec_length = load_list[:, 1].sum() * 12.5 / 1000
            print('%s: # samples: %i / total spec length: %i s (%.2f hr)' %
                  (split.upper(), num_total_sample, total_spec_length, total_spec_length / 60 / 60))

            # split by batch_size
            num_batches_per_epoch = load_list.size(0) // effective_batch_size
            load_list = load_list.view(num_batches_per_epoch, -1, load_list.size(1))

            if sort:
                # sort by text length in each batch (PackedSequence requires it)
                phoneme_len_list = phoneme_len_list[:end_idx].view(num_batches_per_epoch, -1)
                sort_length, sort_idx = phoneme_len_list.sort(1, descending=True)
                sort_idx = sort_idx.view(num_batches_per_epoch, -1, 1).expand_as(load_list)
                load_list = torch.gather(load_list, 1, sort_idx)

            # shuffle by batch (Note that, the order is preserved in each batch)
            if self.shuffle_data:
                _, sort_idx = torch.randn(num_batches_per_epoch).sort()
                sort_idx = sort_idx.view(-1, 1, 1).expand_as(load_list)
                load_list = torch.gather(load_list, 0, sort_idx)  # nbpe x N x 6

            self.load_list[split] = load_list.long()

            # calculate number of iterations per epoch
            spec_lengths = load_list[:, :, 1]      # nbpe x N
            spec_lengths_max, _ = torch.max(spec_lengths, dim=1)    # nbpe
            num_subbatches_per_batch = torch.ceil(spec_lengths_max.float() / self.trunc_size[split])
            self.split_num_iters[split] = int(num_subbatches_per_batch.sum().item())

            # set split cursor
            self.split_sizes[split] = self.load_list[split].size(0)
            self.split_cursor[split] = 0

        # the dictionary `variables` should be defined again to enable pickling in `Process`
        variables = {
            "args": args,
            "load_list": self.load_list,
            "pretraining": self.pretraining,
            "path_dict": self.path_dict,
            "n_workers": self.n_workers,
            "r_factor": self.r_factor,
            "dec_out_size": self.dec_out_size,
            "post_out_size": self.post_out_size,
            "split_sizes": self.split_sizes,
            "use_pitch": self.use_pitch,
            "lang_dict": self.lang_dict
        }
        if hasattr(voxa_config, 'pitch_binary'):
            variables.update({
                "pitch_means": self.pitch_means,
                "pitch_stdevs": self.pitch_stdevs,
                "pitch_max": self.pitch_max,
            })
        for split in self.load_list.keys():
            if self.split_sizes[split] > 0:
                self.process[split] = Process(target=start_async_loader,
                                            args=(variables, split, self.split_cursor[split], self.queue[split]))
                self.process[split].start()

    def next_batch(self, split):
        T, idx = self.trunc_size[split], self.split_cursor[split]

        # seek and load data from raw files
        if self.is_subbatch_end[split]:
            self.is_subbatch_end[split] = False
            self.subbatch_cursor = 0

            self.len_phoneme, self.len_spec, self.curr_phoneme, \
            self.curr_specM, self.curr_specL, self.curr_pitch, \
            self.idx_speaker, self.idx_data, self.idx_lang  = self.queue[split].get()

            self.split_cursor[split] = (idx + 1) % self.split_sizes[split]
            self.subbatch_max_len = self.len_spec.max()
            self.len_spec_original = self.len_spec.clone()

        # Variables to return
        subbatch_len_phoneme = self.len_phoneme.tolist()
        subbatch_len_spec = [min(x.item(), T) for x in self.len_spec]
        x_phoneme = self.curr_phoneme
        y_specM = self.curr_specM[:, self.subbatch_cursor:self.subbatch_cursor + max(subbatch_len_spec)].contiguous()
        y_specM_whole = self.curr_specM            
        y_specL = self.curr_specL[:, self.subbatch_cursor:self.subbatch_cursor + max(subbatch_len_spec)].contiguous()
        y_pitch = self.curr_pitch[:, self.subbatch_cursor:self.subbatch_cursor + max(subbatch_len_spec)].contiguous()
        y_pitch_whole = self.curr_pitch
        idx_speaker = self.idx_speaker
        idx_data = self.idx_data
        idx_lang = self.idx_lang

        if self.use_gpu:
            x_phoneme = x_phoneme.cuda()
            y_specM = y_specM.cuda()
            y_specM_whole = y_specM_whole.cuda() 
            y_specL = y_specL.cuda()
            y_pitch = y_pitch.cuda()
            y_pitch_whole = y_pitch_whole.cuda()
            self.len_spec_original = self.len_spec_original.cuda()
            idx_speaker = idx_speaker.cuda()
            idx_lang = idx_lang.cuda()

        # Advance subbatch_cursor or Move on to the next batch
        if split == 'whole':
            self.is_subbatch_end[split] = True
        else:
            self.subbatch_cursor += T
            if self.subbatch_cursor < self.subbatch_max_len:
                self.len_spec.sub_(T).clamp_(min=0)
            else:
                self.is_subbatch_end[split] = True

        # Don't compute for empty batch elements
        if subbatch_len_spec.count(0) > 0:
            len_spec_mask = [idx for idx, l in enumerate(subbatch_len_spec) if l > 0]

            len_spec_mask = torch.LongTensor(len_spec_mask)
            if self.use_gpu:
                len_spec_mask = len_spec_mask.cuda()

            subbatch_len_phoneme = [subbatch_len_phoneme[idx] for idx in len_spec_mask]
            subbatch_len_spec = [subbatch_len_spec[idx] for idx in len_spec_mask]
            x_phoneme = torch.index_select(x_phoneme, 0, len_spec_mask)[:, :max(subbatch_len_phoneme)]
            y_specM = torch.index_select(y_specM, 0, len_spec_mask)
            y_specM_whole = torch.index_select(y_specM_whole, 0, len_spec_mask)
            y_specL = torch.index_select(y_specL, 0, len_spec_mask)
            y_pitch = torch.index_select(y_pitch, 0, len_spec_mask)
            y_pitch_whole = torch.index_select(y_pitch_whole, 0, len_spec_mask)
            idx_speaker = torch.index_select(idx_speaker, 0, len_spec_mask)
            self.len_spec_original = torch.index_select(self.len_spec_original, 0, len_spec_mask)
            idx_data = [self.idx_data[idx] for idx in len_spec_mask]
            idx_lang = torch.index_select(idx_lang, 0, len_spec_mask)

            if len(len_spec_mask) < len(self.len_mask_mapper):
                # need to do this since decoder_states may be masked before
                len_spec_mask = [self.len_mask_mapper[mask_idx] for mask_idx in len_spec_mask]
                len_spec_mask = torch.LongTensor(len_spec_mask)
                if self.use_gpu:
                    len_spec_mask = len_spec_mask.cuda()

            self.len_mask_mapper = {}
            for i, j in enumerate(len_spec_mask):
                self.len_mask_mapper[j] = i
        else:
            len_spec_mask = None
            self.len_mask_mapper = {}

        return_dict = {
            "x_phoneme": x_phoneme,
            "y_specM": y_specM,
            'y_specM_whole': y_specM_whole,
            "y_specL": y_specL,
            "y_pitch": y_pitch,
            "y_pitch_whole": y_pitch_whole,
            "idx_speaker": idx_speaker,
            "subbatch_len_spec": subbatch_len_spec,
            "subbatch_len_phoneme": subbatch_len_phoneme,
            "len_spec": self.len_spec_original,
            "len_spec_mask": len_spec_mask,
            "idx_data": idx_data,
            "lang_id": idx_lang
        }
        return return_dict        


def start_async_loader(variables, split, load_start_idx, queue):
    # This function should be defined at the top level to enable pickling
    # load batches to the queue asynchronously since it is a bottle-neck
    N = len(variables['load_list'][split][0])
    r = variables['r_factor']
    load_curr_idx = load_start_idx

    feats_to_skip = set([])
    feats_to_skip.add('linear')
    if not variables['use_pitch']:
        feats_to_skip.add('pitch')
    if variables['pretraining'] == 1:
        feats_to_skip.add('linear')
        feats_to_skip.add('phoneme')

    with Pool(variables['n_workers']) as pool:
        while True:
            data_M, data_L, data_P, data_F0, len_M, len_P, spkr_id, data_id, lang_id = ([None for _ in range(N)] for _ in range(9))
            # deploy workers to load data
            partial_func = partial(load_data_and_length,
                                   variables['path_dict'],
                                   variables['load_list'][split][load_curr_idx],
                                   feats_to_skip,
                                   variables['args'],
                                   variables['lang_dict']
                                   )
            results = pool.map_async(func=partial_func, iterable=range(N))

            for result in results.get():
                load_idx = result['load_idx']
                data_M[load_idx] = result['data_M']
                data_L[load_idx] = result['data_L']
                data_P[load_idx] = result['data_P']
                data_F0[load_idx] = result['data_F0']
                len_M[load_idx] = result['len_M']
                len_P[load_idx] = result['len_P']
                spkr_id[load_idx] = result['spkr_id']
                data_id[load_idx] = result['data_id']
                lang_id[load_idx] = result['lang_id']
                if 'pitch_means' in variables:
                    data_F0[load_idx] = normalize_pitch(data_F0[load_idx])

            len_phoneme = torch.IntTensor(len_P)
            len_spec = torch.Tensor(len_M).div(r).ceil().mul(r).int()                       # consider r_factor
            curr_phoneme = torch.LongTensor(N, len_phoneme.max()).fill_(0)                        # null-padding at tail
            curr_specM = torch.Tensor(N, len_spec.max(), variables['dec_out_size']).fill_(0)                 # null-padding at tail
            curr_specL = torch.Tensor(N, len_spec.max(), variables['post_out_size']).fill_(0)                # null-padding at tail
            curr_pitch = torch.Tensor(N, len_spec.max()).fill_(0)                # null-padding at tail
            speaker_id = torch.LongTensor(spkr_id)
            language_id = torch.LongTensor(lang_id)

            # fill the template tensors
            for j in range(N):
                curr_phoneme[j, 0:data_P[j].size(0)].copy_(data_P[j])
                curr_specM[j, 0:data_M[j].size(0)].copy_(data_M[j])
                if variables['post_out_size'] != variables['dec_out_size']:
                    curr_specL[j, 0:data_L[j].size(0)].copy_(data_L[j])
                if 'pitch_means' in variables:
                    curr_pitch[j, 0:data_F0[j].size(0)].copy_(data_F0[j])

            queue.put((len_phoneme, len_spec, curr_phoneme, curr_specM, curr_specL, curr_pitch, speaker_id, data_id, language_id))
            load_curr_idx = (load_curr_idx + 1) % variables['split_sizes'][split]


def load_binary(file_path, offset, length):
    with open(file_path, 'rb') as datafile:
        datafile.seek(offset)
        line = datafile.read(length)
        obj = pickle.loads(line)
    return obj


def load_data_and_length(path_dict, load_list, feats_to_skip, args, lang_dict, load_idx):
    load_list = load_list[load_idx].tolist()
    bin_file_idx = load_list[-1]

    # load mel
    data_M = load_binary(path_dict['mel'][bin_file_idx], load_list[3], load_list[6])
    data_M = torch.from_numpy(data_M)
    len_M = data_M.size(0)

    # load linear
    if not 'linear' in feats_to_skip:
        data_L = load_binary(path_dict['linear'][bin_file_idx], load_list[4], load_list[7])
        data_L = torch.from_numpy(data_L)
    else:
        # dummy
        data_L = torch.Tensor(1, 513)

    # load wav
    if not 'pitch' in feats_to_skip:
        data_F0 = load_binary(path_dict['pitch'][bin_file_idx], load_list[9], load_list[10])
        if type(data_F0) is torch.Tensor:
            pass
        else:
            data_F0 = torch.from_numpy(data_F0)
    else:
        # dummy
        data_F0 = torch.Tensor(1)

    # load phoneme
    if not 'phoneme' in feats_to_skip:
        data_P = load_binary(path_dict['phoneme'][bin_file_idx], load_list[5], load_list[8])
        _, idx_to_vocab = get_sg_vocab()            
        p2l = [phoneme2lang[idx_to_vocab[p]] for p in data_P if idx_to_vocab[p] not in others+symbols]
        lang = max(set(p2l), key=p2l.count) if len(p2l) > 0 else 'ENG'
        if lang in lang_dict:
            lang_id = lang_dict[lang]
        else:
            lang_id = 0
        # Pad space (120: ' ') instead of NULL to stabilize CTC loss
        data_P, commas = pad_text(data_P, model_version=args.model_version, debug=args.debug)
        data_P = torch.LongTensor(data_P)
        len_P = data_P.size(0)
    else:
        # dummy
        data_P = torch.LongTensor(1)
        len_P = 1
        lang_id = torch.LongTensor(1)
        
    # load remaining 
    spkr_id = int(load_list[-2])
    data_idx = int(load_list[0])
    return {
        'load_idx': load_idx,
        'data_M': data_M,
        'data_L': data_L,
        'data_P': data_P,
        'len_M': len_M,
        'len_P': len_P,
        'data_F0': data_F0,
        'spkr_id': spkr_id,
        'data_id': data_idx,
        'lang_id': lang_id
    }
