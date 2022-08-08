import torch
import torch.nn as nn
import torch.nn.functional as F

from tacotron.module.deprecated_model.model_cats_v3.module.fastspeech.Layers import MultiHeadAttention


class MelStyleEncoder(nn.Module):
    def __init__(self, mel_dim, out_dim, d_model, spkr_hidden, debug=0, **kwargs):
        super(MelStyleEncoder, self).__init__()
        n_spectral_layer = 2
        n_temporal_layer = 2
        n_slf_attn_layer = 1
        n_slf_attn_head = 2
        d_k = d_v = d_model // n_slf_attn_head
        dropout = 0.1

        self.fc_1 = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            Mish()
        )
        nn.init.xavier_uniform_(self.fc_1[0].weight)
        nn.init.constant_(self.fc_1[0].bias, 0.0)

        self.spectral_stack = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                Mish()
            )            
            for _ in range(n_spectral_layer)
        ])

        self.temporal_stack = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, 2 * d_model, 5, padding=2),
                Mish(),
                nn.Dropout(dropout),
                nn.GLU(1),
            )
            for _ in range(n_temporal_layer)
        ])

        self.slf_attn_stack = nn.ModuleList([
            MultiHeadAttention(
                n_slf_attn_head, d_model, d_k, d_v, dropout=dropout
            )
            for _ in range(n_slf_attn_layer)
        ])
        self.speaker_biases = nn.Sequential(
            nn.Linear(spkr_hidden, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_slf_attn_layer * n_slf_attn_head * d_model),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(d_model, out_dim),
            Mish()
        )
        nn.init.xavier_uniform_(self.fc_2[0].weight)
        nn.init.constant_(self.fc_2[0].bias, 0.0)

    def forward(self, mel, mask, spkr_vec, debug, **kwargs):
        N, max_len, _ = mel.size()
        ref_out_dict = {}
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        conv_mask = mask.unsqueeze(1)
        try:
            enc_output = enc_output.transpose(1, 2) * conv_mask
        except:
            import pdb; pdb.set_trace()
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = (residual + enc_output) * conv_mask
        enc_output = enc_output.transpose(1, 2)

        # Multi-head self-attention
        spkr_bias_list = torch.chunk(
            self.speaker_biases(spkr_vec),
            len(self.slf_attn_stack) + 1,
            2
        )
        for i, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output + spkr_bias_list[i]

        # Final Layer
        enc_output = self.fc_2(enc_output) # [B, T, H]

        # Temporal Average Pooling
        mask = mask.unsqueeze(2)
        enc_output = torch.sum(enc_output*mask, dim=1, keepdim=True) \
            / torch.sum(mask, dim=1, keepdim=True)          # [B, 1, H]

        ref_out_dict.update({
            'gst': enc_output,
        })
        return ref_out_dict


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MelStyleEncoderIntegrated(nn.Module):
    def __init__(self, mel_dim, out_dim, d_model, spkr_hidden, debug=0, **kwargs):
        super(MelStyleEncoderIntegrated, self).__init__()
        n_spectral_layer = 2
        n_temporal_layer = 4
        n_slf_attn_layer = 1
        n_slf_attn_head = 2
        d_k = d_v = d_model // n_slf_attn_head
        dropout = 0.1

        self.fc_1 = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            Mish()
        )
        nn.init.xavier_uniform_(self.fc_1[0].weight)
        nn.init.constant_(self.fc_1[0].bias, 0.0)

        self.spectral_stack = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                Mish()
            )            
            for _ in range(n_spectral_layer)
        ])

        self.temporal_stack = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, 2 * d_model, 5, padding=2),
                Mish(),
                nn.Dropout(dropout),
                nn.GLU(1),
            )
            for _ in range(n_temporal_layer)
        ])

        self.slf_attn_stack = nn.ModuleList([
            MultiHeadAttention(
                n_slf_attn_head, d_model, d_k, d_v, dropout=dropout
            )
            for _ in range(n_slf_attn_layer)
        ])
        self.speaker_biases = nn.Sequential(
            nn.Linear(spkr_hidden, d_model),
            Mish(),
            nn.Linear(d_model, d_model),
            Mish(),
            nn.Linear(d_model, n_slf_attn_layer * n_slf_attn_head * d_model)
        )

        self.sside_prosody_size = kwargs.get('sside_prosody_size')
        self.fc_finegrained = nn.Linear(d_model, self.sside_prosody_size)
        nn.init.xavier_uniform_(self.fc_finegrained.weight)
        nn.init.constant_(self.fc_finegrained.bias, 0.0)

        self.fc_gst = nn.Sequential(
            nn.Linear(d_model, d_model),
            Mish(),
            nn.Linear(d_model, d_model),
            Mish(),
            nn.Linear(d_model, out_dim),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.fc_gst[0].weight)
        nn.init.constant_(self.fc_gst[0].bias, 0.0)

    def forward(self, mel, mask, spkr_vec, debug, **kwargs):
        N, max_len, _ = mel.size()
        ref_out_dict = {}
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        conv_mask = mask.unsqueeze(1)
        try:
            enc_output = enc_output.transpose(1, 2) * conv_mask
        except:
            import pdb; pdb.set_trace()
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = (residual + enc_output) * conv_mask
        enc_output = enc_output.transpose(1, 2)

        # Multi-head self-attention
        spkr_bias_list = torch.chunk(
            self.speaker_biases(spkr_vec),
            len(self.slf_attn_stack) + 1,
            2
        )
        for i, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output + spkr_bias_list[i]

        mask = mask.unsqueeze(2)
        # Speech-side prosody
        sside_prosody_vec = self.fc_finegrained(enc_output) * mask # [B, T, H]

        # Final Layer
        gst_output = self.fc_gst(enc_output) # [B, T, H]

        # Global style vector (Temporal Average Pooling)
        gst_output = torch.sum(gst_output*mask, dim=1, keepdim=True) \
            / torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=1)          # [B, 1, H]

        ref_out_dict.update({
            'gst': gst_output,
            'sside_prosody_vec': sside_prosody_vec,
        })
        return ref_out_dict


class SpeechEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use the last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
