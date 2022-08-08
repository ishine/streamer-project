import torch
import torch.nn as nn


class ProsodyStatsGST(nn.Module):
    """
    Store some statistics of each speaker (ex. prosody, speed)
    Currently available statistics:
        - mean: mean prosody of normal sentence
        - speed: mean of voice speed
    """
    def __init__(self, num_spkr, prosody_dim, vocab, debug=0):
        super(ProsodyStatsGST, self).__init__()
        self.num_spkr = num_spkr
        self.stat_types = ['means', 'question',]
        self.register_buffer('max_pitch', torch.zeros(num_spkr))
        self.register_buffer('speed', torch.zeros(num_spkr, 1))
        self.register_buffer('means', torch.zeros(num_spkr, prosody_dim))
        self.register_buffer('question', torch.zeros(num_spkr, prosody_dim))        # deprecated from cats4
        self.vocab = vocab

        self.acc_count = {}
        self.acc_mean = {}
        self.raw_list = {}
        self.reset_acc_states()

    def put_stats(self, spkr_id, prosody_vec, speed=None, phoneme_input=None):
        for i, s in enumerate(spkr_id):
            self.acc_mean['means'][s] += prosody_vec[i]
            self.acc_count['means'][s] += 1
            self.raw_list['speed'][s].append(speed[i].item())

    def summarize_acc_states(self):
        for s in range(self.num_spkr):
            if self.acc_count['means'][s] > 0:
                self.means[s].copy_(self.acc_mean['means'][s] / self.acc_count['means'][s])

            speed_list = self.raw_list['speed'][s]
            if len(speed_list) > 0:
                self.speed[s].fill_(sorted(speed_list)[int(len(speed_list)//2)])
        self.reset_acc_states()

    def reset_acc_states(self):
        for stat_type in self.stat_types:
            self.acc_count[stat_type] = [0 for _ in range(self.num_spkr)]
            self.acc_mean[stat_type] = [0 for _ in range(self.num_spkr)]

        self.raw_list['speed'] = [[] for _ in range(self.num_spkr)]

    def put_max_pitch(self, spkr_id, pitch_value):
        self.max_pitch[spkr_id].fill_(pitch_value)
