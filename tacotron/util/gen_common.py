import os
import torch
import numpy as np
from collections import namedtuple

from tts_text_util.get_vocab import get_sg_vocab, get_phoneme_vocab
from voxa import VoxaConfig
from voxa.speakermgr.speaker_manager import SpeakerManager

from tacotron.util.default_args import load_and_override_args
from tacotron.util.model_load import support_deprecated_model_gst
Loaded_info = namedtuple('SeqProsodyModel', 'model args checkpoint voxa_config data_list speaker_manager lang_dict')
BACKWARD_COMPATIBLE_VOXA_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets/voxa_config_default_1912.yaml')

def load_model_common(new_args, ignore_attention=True, **kwargs):
    args, checkpoint = get_args_checkpoint(new_args, **kwargs)
    model = load_tacotron(args, checkpoint, ignore_attention)
    voxa_config = set_voxa_config(new_args, checkpoint)
    data_list = sorted(args.data.split(','))
    speaker_manager = SpeakerManager(data_list)
    lang_dict = checkpoint['lang_dict'] if 'lang_dict' in checkpoint else None
    return Loaded_info(model, args, checkpoint, voxa_config, data_list, speaker_manager, lang_dict)

def get_args_checkpoint(new_args, **kwargs):
    # make args as dictionary
    new_args = vars(new_args)

    # load and override some arguments
    new_args.update(kwargs)
    checkpoint = torch.load(new_args['init_from'], map_location=lambda storage, loc: storage)
    args = load_and_override_args(checkpoint['args'], new_args)
    args.gpu = new_args['gpu']
    return args, checkpoint

def auto_clamp(pred_spec, sp, dec_out_type):
    if sp.use_spectrogram_clipping:
        pred_spec = torch.clamp(pred_spec, 0, 1)
    else:
        if dec_out_type == 'lin':
            if sp.apply_pre_emphasis:
                pred_spec = torch.clamp(pred_spec, -11.51, 2.85)  # clamp outlier of linear spectrogram w/o clipping. [-11.51, 2.85] is possible range
            else:
                pred_spec = torch.clamp(pred_spec, -11.51, 4.29)  # clamp outlier of linear spectrogram w/o clipping and preemphasis. [-11.51, 4.29] is possible range
        else:
            # not implemented
            pass
    return pred_spec

def load_tacotron(args, checkpoint, ignore_attention=True):
    model_version = args.model_version
    if args.gst == 1:
        Tacotron = support_deprecated_model_gst(args.exp_no, model_version).Tacotron
    elif args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron
    else:
        raise RuntimeError('Please specify valid args.gst')

    if args.use_sg:
        vocab, idx_to_vocab = get_sg_vocab()
    else: 
        vocab, idx_to_vocab = get_phoneme_vocab()
    model = Tacotron(args, vocab=vocab, idx_to_vocab=idx_to_vocab)

    # ignore attention module in checkpoint when only the alignment algorithm is changed.
    state_dict = checkpoint['state_dict']
    if ignore_attention:
        for key in model.state_dict().keys():
            if key.startswith('attention'):
                state_dict[key] = model.state_dict()[key]
    keys_to_pop = []
    for key in state_dict:
        if not key in model.state_dict().keys():
            keys_to_pop.append(key)
    for key in keys_to_pop:
        state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.reset_decoder_states(debug=args.debug)
    print('loaded checkpoint %s' % (args.init_from))
    model = model.eval()

    if args.gpu is None or len(args.gpu) == 0:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
            model = model.cuda()
        else:
            pass
            # raise NotImplementedError()
    return model


def set_voxa_config(new_args, checkpoint):
    # make args as dictionary
    new_args = vars(new_args)

    # set voxa config
    new_voxa_config = new_args.get('config', None)
    if new_voxa_config is None:
        if 'config' in checkpoint:
            voxa_config = VoxaConfig.load_from_config(checkpoint['config'], initialize=False)
        else:   
            voxa_config = VoxaConfig(BACKWARD_COMPATIBLE_VOXA_CFG, initialize=False)
    else:
        voxa_config = VoxaConfig(new_voxa_config)
    return voxa_config


def gen_output_full_path(args, cleansed):
    if hasattr(args, 'out_files'):
        return args.out_files['wav'], args.out_files['attention'], args.out_files['mel']
    else:
        if args.outfile:
            outpath1 = args.outfile
        else:
            file_name = cleansed[:20]
            outpath1 = '%s/%s_%s.wav' % (args.out_dir, args.exp_no, file_name)

        # generate director
        if not os.path.exists(os.path.dirname(outpath1)):
            os.makedirs(os.path.dirname(outpath1), mode=0o755, exist_ok=True)
        base_name, _ = os.path.splitext(outpath1)
        return outpath1, f'{base_name}.png', f'{base_name}.{args.target_type}.npy'


def gen_prosody_output_full_path(args, n):
    if hasattr(args, 'out_files'):
        return args.out_files['patt_img'], args.out_files['patt_meta']
    else:
        if args.outfile:
            outpath1 = args.outfile
        else:
            file_name = args.caption[:20]
            outpath1 = '%s/%s_%s_%s.wav' % (args.out_dir, args.exp_no, file_name, n)

        # generate director
        if not os.path.exists(os.path.dirname(outpath1)):
            os.makedirs(os.path.dirname(outpath1), mode=0o755, exist_ok=True)
        base_name, _ = os.path.splitext(outpath1)
        return f'{base_name}_patt.png', f'{base_name}_prosody.json'


def plot_attention(input_sentence, attentions, outpath):
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.cpu().numpy(), aspect='auto', origin='upper',cmap='gray')
    # fig.colorbar(cax)
    plt.ylabel('Encoder timestep', fontsize=18)
    plt.xlabel('Decoder timestep', fontsize=18)

    if input_sentence:
        plt.ylabel('Encoder timestep', fontsize=18)
        # Set up axes
        # ax.set_yticklabels([' '] + list(input_sentence) + [' '])
        # Show label at every tick
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def preprocess_audio(path, signal_processing):
    audio_padded, _ = signal_processing.file2padded_waveform(path)
    mel_spec = signal_processing.spectrogram(audio_padded, type='mel', input_type='waveform', is_numpy=False)
    return mel_spec.float().unsqueeze(0)


def get_pitch_from_audio(path, signal_processing, use_cuda=False):
    audio_padded, _ = signal_processing.file2padded_waveform(path)
    pitch = signal_processing.estimate_pitch(audio_padded.float(), use_cuda=use_cuda)
    return normalize_pitch(pitch.unsqueeze(0))


def plot_spec(spectrogram, outpath, signal_processing):
    """
    :param spectrogram: spectrogram ndarray of size C x T
    """
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display
    librosa.display.specshow(spectrogram, x_axis='time', sr=signal_processing.sample_rate,
                             vmax=1.0, vmin=0.0, cmap=plt.cm.inferno)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def trim_and_add_silence(spectrogram, frame_shift_inMS=12.5):
    """
    trim silence from the beginning of spectrogram
    :param spectrogram: torch.Tensor (T x H)
    :param frame_shift_inMS: frame_shift_inMS defined in VoxaConfig.
    :return: trimmed spectrogram
    """
    T = spectrogram.shape[0]
    n_silence_allowed = int(50 / frame_shift_inMS)

    # Denormalize & Convert back to linear
    spec = (np.clip(spectrogram, 0, 1) * 100) - 100
    spec = np.power(10.0, spec * 0.05)

    spec_norm = np.linalg.norm(spec, axis=1)
    min_norm = min(spec_norm)
    max_norm = max(spec_norm)
    buffer = (max_norm - min_norm) * 0.01

    start_idx = 0
    start_idx_limit = int(1000 / frame_shift_inMS)
    for i in range(start_idx_limit):
        if spec_norm[i] > min_norm + buffer:
            break
        start_idx += 1

    # leave at least 50ms of silence
    start_idx -= n_silence_allowed

    # ensure validity
    start_idx = min(max(start_idx, 0), T)

    spec_trimmed = spectrogram[start_idx:]

    # add trailing silence
    output = np.concatenate([spec_trimmed, spec_trimmed[:n_silence_allowed].copy()*0], axis=0)
    return output


def trim_and_add_silence(spectrogram, frame_shift_inMS=12.5):
    """
    trim silence from the beginning of spectrogram
    :param spectrogram: torch.Tensor (T x H)
    :param frame_shift_inMS: frame_shift_inMS defined in VoxaConfig.
    :return: trimmed spectrogram
    """
    T = spectrogram.shape[0]
    n_silence_allowed = int(50 / frame_shift_inMS)

    # Denormalize & Convert back to linear
    spec = (np.clip(spectrogram, 0, 1) * 100) - 100
    spec = np.power(10.0, spec * 0.05)

    spec_norm = np.linalg.norm(spec, axis=1)
    min_norm = min(spec_norm)
    max_norm = max(spec_norm)
    buffer = (max_norm - min_norm) * 0.01

    start_idx = 0
    start_idx_limit = int(1000 / frame_shift_inMS)
    for i in range(start_idx_limit):
        if spec_norm[i] > min_norm + buffer:
            break
        start_idx += 1

    # leave at least 50ms of silence
    start_idx -= n_silence_allowed

    # ensure validity
    start_idx = min(max(start_idx, 0), T)

    spec_trimmed = spectrogram[start_idx:]

    # add trailing silence
    output = np.concatenate([spec_trimmed, spec_trimmed[:n_silence_allowed].copy()*0], axis=0)
    return output


def check_remove_phoneme(voxa_config):
    if hasattr(voxa_config, 'rm_dummy_ph') and voxa_config.rm_dummy_ph:
        return True
    else:
        return False

vocab_to_idx, _ = get_sg_vocab()
plosives = set([
    vocab_to_idx[x]
    for x in ["B", "CH", "D", "G", "JH", "K", "P", "T", "g_ko", "d_ko", "b_ko", "x_ko", "w_ko", "f_ko"]
])
diph_converter = {
    vocab_to_idx["AW0"]: [vocab_to_idx["AW0"], vocab_to_idx["UH0"]],
    vocab_to_idx["AW1"]: [vocab_to_idx["AW1"], vocab_to_idx["UH1"]],
    vocab_to_idx["AW2"]: [vocab_to_idx["AW2"], vocab_to_idx["UH2"]],
    vocab_to_idx["AY0"]: [vocab_to_idx["AW0"], vocab_to_idx["IH0"]],
    vocab_to_idx["AY1"]: [vocab_to_idx["AW1"], vocab_to_idx["IH1"]],
    vocab_to_idx["AY2"]: [vocab_to_idx["AW2"], vocab_to_idx["IH2"]],
    vocab_to_idx["EY0"]: [vocab_to_idx["EY0"], vocab_to_idx["IH0"]],
    vocab_to_idx["EY1"]: [vocab_to_idx["EY1"], vocab_to_idx["IH1"]],
    vocab_to_idx["EY2"]: [vocab_to_idx["EY2"], vocab_to_idx["IH2"]],
    vocab_to_idx["OW0"]: [vocab_to_idx["OW0"], vocab_to_idx["UH0"]],
    vocab_to_idx["OW1"]: [vocab_to_idx["OW1"], vocab_to_idx["UH1"]],
    vocab_to_idx["OW2"]: [vocab_to_idx["OW2"], vocab_to_idx["UH2"]],
    vocab_to_idx["OY0"]: [vocab_to_idx["AO0"], vocab_to_idx["IH0"]],
    vocab_to_idx["OY1"]: [vocab_to_idx["AO1"], vocab_to_idx["IH1"]],
    vocab_to_idx["OY2"]: [vocab_to_idx["AO2"], vocab_to_idx["IH2"]],
}
stress_dict = {
    vocab_to_idx["AA1"]: vocab_to_idx["AA0"],
    vocab_to_idx["AA2"]: vocab_to_idx["AA0"],
    vocab_to_idx["AE1"]: vocab_to_idx["AE0"],
    vocab_to_idx["AE2"]: vocab_to_idx["AE0"],
    vocab_to_idx["AH1"]: vocab_to_idx["AH0"],
    vocab_to_idx["AH2"]: vocab_to_idx["AH0"],
    vocab_to_idx["AO1"]: vocab_to_idx["AO0"],
    vocab_to_idx["AO2"]: vocab_to_idx["AO0"],
    vocab_to_idx["AW1"]: vocab_to_idx["AW0"],
    vocab_to_idx["AW2"]: vocab_to_idx["AW0"],
    vocab_to_idx["AY1"]: vocab_to_idx["AY0"],
    vocab_to_idx["AY2"]: vocab_to_idx["AY0"],
    vocab_to_idx["EH1"]: vocab_to_idx["EH0"],
    vocab_to_idx["EH2"]: vocab_to_idx["EH0"],
    vocab_to_idx["ER1"]: vocab_to_idx["ER0"],
    vocab_to_idx["ER2"]: vocab_to_idx["ER0"],
    vocab_to_idx["EY1"]: vocab_to_idx["EY0"],
    vocab_to_idx["EY2"]: vocab_to_idx["EY0"],
    vocab_to_idx["IH1"]: vocab_to_idx["IH0"],
    vocab_to_idx["IH2"]: vocab_to_idx["IH0"],
    vocab_to_idx["IY1"]: vocab_to_idx["IY0"],
    vocab_to_idx["IY2"]: vocab_to_idx["IY0"],
    vocab_to_idx["OW1"]: vocab_to_idx["OW0"],
    vocab_to_idx["OW2"]: vocab_to_idx["OW0"],
    vocab_to_idx["OY1"]: vocab_to_idx["OY0"],
    vocab_to_idx["OY2"]: vocab_to_idx["OY0"],
    vocab_to_idx["UH1"]: vocab_to_idx["UH0"],
    vocab_to_idx["UH2"]: vocab_to_idx["UH0"],
    vocab_to_idx["UW1"]: vocab_to_idx["UW0"],
    vocab_to_idx["UW2"]: vocab_to_idx["UW0"],
}
def pad_text(lst, null_padding=None, model_version=None, debug=0):
    comma_position = []
    commas = None
    if model_version is None:
        result = [null_padding] + lst + [null_padding]
    else:
        if model_version.startswith('cats'):
            if model_version == 'cats4':
                new_data_P = []
                for i, x in enumerate(lst):
                    # remove all commas (TODO: Do the same thing for '?', '!', '.'?)
                    if x == vocab_to_idx[',']:
                        comma_position.append(len(new_data_P))
                        continue
                    
                    # add space before plosives
                    if x in plosives and i > 1:
                        new_data_P.append(vocab_to_idx[' '])
                    
                    # degenerate stress
                    if x in stress_dict:
                        x = stress_dict[x]

                    # convert diphthong as two phonemes
                    if x in diph_converter:
                        new_data_P.extend(diph_converter[x])
                    else:
                        new_data_P.append(x)
                lst = new_data_P
                commas = [0 for _ in range(len(lst))]
                for i in comma_position:
                    if i < len(commas):
                        commas[i] = 1
                commas = [0] + commas + [0]
            result = [vocab_to_idx[' ']] + lst + [vocab_to_idx[' ']]
        elif model_version == 'align':
            result = [0] + lst + [0]
        else:
            result = lst + [0]
    return result, commas


def normalize_pitch(pitch, max_val=880.0):
    # 880 is max_pitch value in voxa preprocessing
    return pitch / max_val
