# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os
num_core = os.cpu_count()
if num_core == 20:
    os.putenv("OMP_NUM_THREADS", str(num_core // 4))
else:
    os.putenv("OMP_NUM_THREADS", str(num_core // 2))

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Currently under experiments:
#
# In model.py | Tacotron...


def verify_args(args):
    # this resolution scale is fixed in the model definition.
    assert args.spec_limit % args.r_factor == 0
    assert args.trunc_size % args.r_factor == 0

# def update_args(path_to_model, default_args):
#     checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
#     new_args = checkpoint['args']
#     for key in default_args.keys():
#         if not key in new_args.__dict__:
#             new_args.__dict__[key] = default_args[key]
#
#     checkpoint['args'] = new_args

def adapt_from_ckpt(new_state, old_state):
    return torch.mean(old_state, dim=0, keepdim=True).expand_as(new_state)

def get_adapted_phoneme_embedding_matrix(len_vocab, old_embedding_matrix, key):
    if key == 'attention.ctc_proj.bias': 
        new_embedding_matrix = torch.mean(old_embedding_matrix, dim=0).repeat((len_vocab))
    else:
        emb_dtype = type(old_embedding_matrix[0].sum().item())
        if emb_dtype is int:
            new_embedding_matrix = torch.zeros(len_vocab, 1, dtype=old_embedding_matrix.dtype)
        else:
            new_embedding_matrix = torch.mean(old_embedding_matrix, dim=0).repeat((len_vocab, 1))
    
    len_emb_matrix = min(old_embedding_matrix.size(0), new_embedding_matrix.size(0))
    new_embedding_matrix[0:len_emb_matrix] = old_embedding_matrix[:len_emb_matrix]

    return new_embedding_matrix

def decay_learning_rate(init_lr, iteration):
    warmup_threshold = 4000
    step = iteration + 1
    decayed_lr = init_lr * warmup_threshold ** 0.5 * min(step * warmup_threshold ** -1.5, step ** -0.5)
    return decayed_lr


def set_default_GST(loader, model, args):
    with torch.no_grad():
        for k in range(loader.split_sizes['whole']):
            loader_dict = loader.next_batch('whole')
            phoneme_input = loader_dict.get("x_phoneme")
            ref_target_mel = loader_dict.get("y_specM_whole")
            spkr_id = loader_dict.get("idx_speaker")
            spec_lengths = loader_dict.get("len_spec")
            phoneme_lengths = loader_dict.get("subbatch_len_phoneme")
            spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S            

            whole_spec_mask = torch.arange(0, spec_lengths.max(), device=spec_lengths.device).view(1, -1).expand(spec_lengths.size(0), -1)
            whole_spec_mask = torch.lt(whole_spec_mask, spec_lengths.view(-1, 1).expand(-1, spec_lengths.max()))             # N x T_dec
            tmp_prosody = model.ref_encoder(
                ref_target_mel,
                whole_spec_mask,
                spkr_vec,
                debug=args.debug
            )['gst']
            phoneme_lengths = torch.tensor(phoneme_lengths, device=phoneme_input.device)         
            speed = phoneme_lengths.type_as(spkr_vec) / spec_lengths.type_as(spkr_vec)
            model.prosody_stats.put_stats(
                spkr_id,
                tmp_prosody.squeeze(1),
                speed=speed,
                phoneme_input=phoneme_input
            )
        model.prosody_stats.summarize_acc_states()

class exponential_moving_average(object):
    def __init__(self, decay=0.9999):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time, math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
