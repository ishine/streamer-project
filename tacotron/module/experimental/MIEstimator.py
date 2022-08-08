import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MIEstimator(nn.Module):
    def __init__(self, vocab_size, decoder_dim, hidden_size, dropout=0.5):
        super(MIEstimator, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(decoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, vocab_size + 1),
        )
        self.ctc = nn.CTCLoss(blank=vocab_size, reduction='none')

    def forward(self, decoder_outputs, target_phones, decoder_lengths, target_lengths):
        log_probs = self.proj(decoder_outputs).log_softmax(dim=2)
        log_probs = log_probs.transpose(1, 0)
        ctc_loss = self.ctc(log_probs, target_phones, decoder_lengths, target_lengths)
        # average by number of frames since taco_loss is averaged.
        ctc_loss = (ctc_loss / decoder_lengths.float()).mean()
        return ctc_loss
