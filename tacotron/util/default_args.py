import argparse, sys, yaml


def get_default_training_args(parse=True):
    args_dict = [
        # data load
        {"name": "data", "type": str, "default": None, "nargs": None, "help": 'dataset to use. comma seperate to use multiple dataset'},
        {"name": "ft_lang", "type": str, "default": '', "nargs": None, "help": 'language specification for fine-tuning'},
        {"name": "ft_gender", "type": str, "default": '', "nargs": None, "help": 'gender specification for fine-tuning'}, 
        {"name": "config", "type": str, "default": None, "nargs": None, "help": 'path to voxa config'},
        {"name": "use_sg", "type": int, "default": 1, "nargs": None, "help": 'use sg phoneme'},
        {"name": "batch_size", "type": int, "default": 32, "nargs": None, "help": 'batch size'},
        {"name": "text_limit", "type": int, "default": 192, "nargs": None, "help": 'maximum length of text to include in training set'},
        {"name": "spec_limit", "type": int, "default": 1000, "nargs": None, "help": 'maximum length of spectrogram to include in training set'},
        {"name": "trunc_size", "type": int, "default": 500, "nargs": None, "help": 'used for truncated-BPTT when memory is not enough.'},
        {"name": "shuffle_data", "type": int, "default": 1, "nargs": None, "help": 'whether to shuffle data loader'},
        {"name": "load_queue_size", "type": int, "default": 4, "nargs": None, "help": 'maximum number of batches to load on the memory'},
        {"name": "n_workers", "type": int, "default": 2, "nargs": None, "help": 'number of workers used in data loader'},
        {"name": "no_validation", "type": int, "default": 0, "nargs": None, "help": '1 to use whole data for training'},
        # model
        {"name": "charvec_dim", "type": int, "default": 512, "nargs": None, "help": ''},
        {"name": "enc_hidden", "type": int, "default": 512, "nargs": None, "help": ''},
        {"name": "jpn_acc_dim", "type": int, "default": 32, "nargs": None, "help": ''},
        {"name": "att_hidden", "type": int, "default": 128, "nargs": None, "help": ''},
        {"name": "dec_hidden", "type": int, "default": 256, "nargs": None, "help": ''},
        {"name": "dec_out_type", "type": str, "default": 'mel', "nargs": None, "help": 'output type: lin/mel'},
        {"name": "spkr_embed_size", "type": int, "default": 64, "nargs": None, "help": 'speaker embedding dimension'},
        {"name": "prosody_size", "type": int, "default": 2, "nargs": None, "help": 'prosody embedding dimension'},
        {"name": "key_size", "type": int, "default": 512, "nargs": None, "help": 'dimension of prosody attention key'},
        {"name": "context_hidden_size", "type": int, "default": 512, "nargs": None, "help": 'prosody embedding dimension'},
        {"name": "num_trans_layer", "type": int, "default": 2, "nargs": None, "help": 'number of layers in transformer part'},
        {"name": "r_factor", "type": int, "default": 1, "nargs": None, "help": 'reduction factor(# of multiple output)'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "fluency", "type": int, "default": 0, "nargs": None, "help": '1 to use fluent model, 0 to use non-fluent model'},
        {"name": "dropout", "type": float, "default": 0.5, "nargs": None, "help": ''},
        # optimization
        {"name": "max_epochs", "type": int, "default": 300, "nargs": None, "help": 'maximum epoch to train'},
        {"name": "max_iters", "type": int, "default": -1, "nargs": None, "help": 'maximum iteration to train (approximate)'},
        {"name": "grad_clip", "type": float, "default": 1, "nargs": None, "help": 'gradient clipping'},
        {"name": "learning_rate", "type": float, "default": 1e-3, "nargs": None, "help": '2e-3 from Ito, I used to use 5e-4'},
        {"name": "lr_decay", "type": int, "default": 1, "nargs": None, "help": '1 to use learning rate decay'},
        {"name": "prosody_recon", "type": int, "default": 0.01, "nargs": None, "help": '1 to use learning rate decay'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 1, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "aug_teacher_forcing", "type": int, "default": 1, "nargs": None, "help": '1 to feed generated output as input to make model robust'},
        {"name": "train_durpred", "type": int, "default": 0, "nargs": None, "help": '1 to jointly training a duration predictor'},
        {"name": "spectral_flatness", "type": int, "default": 1, "nargs": None, "help": '1 to consider spectral flatness in loss calculation'},
        {"name": "context_weight", "type": float, "default": 1, "nargs": None, "help": ''},
        {"name": "adv_weight", "type": float, "default": 0.0005, "nargs": None, "help": ''},
        # loading
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "aligner_from", "type": str, "default": '', "nargs": None, "help": 'load aligner from...'},
        {"name": "filter_data_file", "type": str, "default": 'loss_summary/filtered_samples_dict.t7', "nargs": None, "help": 'filter out bad examples using...'},
        {"name": "freeze", "type": str, "default": '', "nargs": None, "help": 'freeze params. comma seperate to freeze multiple modules.'},
        {"name": "balanced_spkr", "type": int, "default": 0, "nargs": None, "help": None},
        {"name": "resume", "type": int, "default": 0, "nargs": None, "help": '1 for resume from saved epoch'},
        # misc
        {"name": "exp_no", "type": str, "default": '0', "nargs": None, "help": ''},
        {"name": "pretraining", "type": int, "default": 0, "nargs": None, "help": 'set this to 1 for pretraining model'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "print_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "save_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "veripe_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "save_dir", "type": str, "default": '/nas/yg/data/Tacotron2-seqprosody/checkpoint', "nargs": None, "help": ''},
        {"name": "model_version", "type": str, "default": 'cats4', "nargs": None, "help": 'used for distinguishing code compatibility'},
        {"name": "mem_prealloc", "type": int, "default": 0, "nargs": None, "help": '1 to use memory preallocation to avoid OOM error'},
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "eval_set", "type": str, "default": 'default', "nargs": None, "help": 'evaluation set'},
        # debug
        {"name": "debug", "type": int, "default": 0, "nargs": None, "help": '1 for debug mode'},
        {"name": "conv", "type": int, "default": 0, "nargs": None, "help": '1 to use conv encoder.'},
        {"name": "gst", "type": int, "default": 1, "nargs": None, "help": '0-fine-grained;1-gst;'},
        {"name": "n_token", "type": int, "default": 10, "nargs": None, "help": 'number of style token'},
        {"name": "n_head", "type": int, "default": 0, "nargs": None, "help": 'positive int to use multi-head attention'},
        {"name": "beta", "type": float, "default": 1, "nargs": None, "help": 'beta of beta-VAE'},
        {"name": "prosody_ignore_spkrs", "type": str, "default": None, "nargs": None, "help": 'speakers to ignore prosody prediction (e.g. singing voice)'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict


def get_default_synthesis_args(parse=True):
    args_dict = [
        # path option
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "out_dir", "type": str, "default": 'generated/', "nargs": None, "help": ''},
        {"name": "ref_txt", "type": str, "default": '', "nargs": None, "help": 'path to reference text file'},
        {"name": "outfile", "type": None, "default": '', "nargs": None, "help": None},
        # text option
        {"name": "target_lang", "type": str, "default": 'KOR', "nargs": None, "help": 'KOR, ENG, None'},
        {"name": "remove_dummy_pho", "type": int, "default": 1, "nargs": None, "help": '1 to Remove dummy phonemes in g2p'},
        {"name": "caption", "type": str, "default": '', "nargs": None, "help": 'text to generate speech'},
        # speaker option
        {"name": "tgt_spkr", "type": str, "default": '', "nargs": None, "help": 'joined form of speaker id to force'},
        {"name": "morph_spkrs", "type": str, "default": '', "nargs": None, "help": 'joined form of speaker id to force'},
        {"name": "morph_ratio", "type": str, "default": '', "nargs": None, "help": ''},
        # gst option
        {"name": "gst_source", "type": str, "default": 'gst_mean', "nargs": None, "help": 'gst_mean/cluster/ref_wav'},
        {"name": "gst_spkr", "type": str, "default": '', "nargs": None, "help": 'speaker for setting gst'},
        {"name": "gst_idx", "type": int, "default": 0, "nargs": None, "help": 'style index'},
        {"name": "gst_ref_wav", "type": str, "default": '', "nargs": None, "help": 'path to reference wav file'},
        # fine-grained prosody option
        {"name": "prosody_source", "type": str, "default": 'prediction', "nargs": None, "help": 'prediction/prosody_vec/ref_wav'},
        {"name": "prosody_spkr", "type": str, "default": '', "nargs": None, "help": 'speaker for setting gst'},
        {"name": "prosody_ref_file", "type": str, "default": '', "nargs": None, "help": 'path to reference prosody file'},
        {"name": "last_pitch_level", "type": float, "default": None, "nargs": None, "help": 'path to reference prosody file'},
        # {"name": "prosody_ref_wav", "type": str, "default": '', "nargs": None, "help": 'the same gst_ref_wav is used for fine-grained prosody.'},
        # generation option
        {"name": "batch_size", "type": int, "default": 16, "nargs": None, "help": 'batch size'},
        {"name": "wave_limit", "type": int, "default": 1000, "nargs": None, "help": 'force generated wave length'},
        {"name": "speed", "type": float, "default": None, "nargs": None, "help": 'absoulte value of voice speed.'},
        {"name": "speed_x", "type": float, "default": 1.0, "nargs": None, "help": 'multiply voice speed by this value.'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "target_type", "type": str, "default": 'wav', "nargs": None, "help": 'output type'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "num_recon_iters", "type": int, "default": 50, "nargs": None, "help": '# of iteration in griffin-lim recon'},
        # misc
        {"name": "exp_no", "type": str, "default": '0', "nargs": None, "help": ''},
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 0, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "no_vocoder", "type": int, "default": 0, "nargs": None, "help": 'only generate mel spectrogram'},
        {"name": "spec_pow", "type": float, "default": 1.0, "nargs": None, "help": 'power value for spectrum input in griffin-lim'},
        {"name": "enable_loudness", "type": int, "default": 0, "nargs": None, "help": 'enable loudness'},
        {"name": "enable_after_effect", "type": int, "default": 0, "nargs": None, "help": 'enable after effect such as laugh'},
        {"name": "ignore_version_conflict", "type": int, "default": 0, "nargs": None, "help": 'set 1 to ignore version conflict error'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict

def get_default_synthesized_bin_args(parse=True):
    args_dict = [
        # path option
        {"name": "config", "type": str, "default": None, "nargs": None, "help": 'path to voxa config'},
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "out_dir", "type": str, "default": 'generated/', "nargs": None, "help": ''},
        {"name": "ref_spkr", "type": str, "default": '', "nargs": None, "help": 'path to reference wav file'},
        {"name": "ref_txt", "type": str, "default": '', "nargs": None, "help": 'path to reference text file'},
        {"name": "ref_wav", "type": str, "default": '', "nargs": None, "help": 'path to reference wav file'},
        {"name": "prosody_from", "type": str, "default": '', "nargs": None, "help": 'path to reference prosody file'},
        {"name": "outfile", "type": None, "default": '', "nargs": None, "help": None},
        {"name": "morph_spkrs", "type": str, "default": '', "nargs": None, "help": 'joined form of speaker id to force'},
        {"name": "morph_ratio", "type": str, "default": '', "nargs": None, "help": ''},
        # text option
        {"name": "target_lang", "type": str, "default": 'KOR', "nargs": None, "help": 'KOR, ENG, None'},
        {"name": "caption", "type": str, "default": '', "nargs": None, "help": 'text to generate speech'},
        # style/speaker option
        {"name": "force_speaker", "type": str, "default": '', "nargs": None, "help": 'joined form of speaker id to force'},
        {"name": "whitening_coeff", "type": float, "default": 1.0, "nargs": None, "help": 'coefficient for whitening (can control pitch)'},
        {"name": "target_type", "type": str, "default": 'wav', "nargs": None, "help": 'output type'},
        {"name": "style_source", "type": str, "default": 'gst_mean', "nargs": None, "help": 'gst_mean/cluster/prosody_vector/ref_wav'},
        {"name": "style_idx", "type": int, "default": 0, "nargs": None, "help": 'style index'},
        {"name": "speed_x", "type": float, "default": 1.0, "nargs": None, "help": 'multiply voice speed by this value.'},
        # generation option
        {"name": "batch_size", "type": int, "default": 16, "nargs": None, "help": 'batch size'},
        {"name": "wave_limit", "type": int, "default": 1000, "nargs": None, "help": 'force generated wave length'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "num_recon_iters", "type": int, "default": 50, "nargs": None, "help": '# of iteration in griffin-lim recon'},
        # misc
        {"name": "exp_no", "type": str, "default": '0', "nargs": None, "help": ''},
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 0, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "no_vocoder", "type": int, "default": 0, "nargs": None, "help": 'only generate mel spectrogram'},
        {"name": "spec_pow", "type": float, "default": 1.0, "nargs": None, "help": 'power value for spectrum input in griffin-lim'},
        {"name": "enable_loudness", "type": int, "default": 0, "nargs": None, "help": 'enable loudness'},
        {"name": "enable_after_effect", "type": int, "default": 0, "nargs": None, "help": 'enable after effect such as laugh'},
        {"name": "only_spkrvec", "type": int, "default": 0, "nargs": None, "help": 'enable after effect such as laugh'},
        {"name": "output_dir", "type": str, "default": 'spkrvec', "nargs": None, "help": ''},
        {"name": "gender_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        {"name": "mage_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        {"name": "fage_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        {"name": "strength_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        {"name": "tone_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        {"name": "tone_down_ratio", "type": float, "default": 0.0, "nargs": None, "help": ''},
        # adding for syntehsized_bin
        {"name": "data", "type": str, "default": '', "nargs": None, "help": 'dataset to use. comma seperate to use multiple dataset'},
        {"name": "feature", "type": str, "default": '', "nargs": None, "help": 'comma seperated features to binarize: {", ".join(valid_feats)'},
        {"name": "use_sg", "type": int, "default": 1, "nargs": None, "help": 'use sg phoneme'},
        {"name": "global_white_list", "type": int, "default": 0, "nargs": None, "help": 'lobally applicable whitelist'},
        {"name": "global_black_list", "type": int, "default": 0, "nargs": None, "help": 'globally applicable blacklist'},
        {"name": "taco_from", "type": str, "default": '', "nargs": None, "help": 'path to tacotron checkpoint'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict

def get_default_evaluation_args(parse=True):
    args_dict = [
        # path option
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "text_from", "type": str, "default": '', "nargs": None, "help": 'text file to generate speech. Separated by lf.'},
        {"name": "out_dir", "type": str, "default": 'generated', "nargs": None, "help": ''},
        {"name": "eval_set", "type": str, "default": 'default', "nargs": None, "help": 'evaluation set from mdl_spkr_txt_pairs_from'},
        {"name": "mdl_spkr_txt_pairs_from", "type": str, "default": 'tacotron/assets/eval_mdl_spkr_txt_pair.yaml', "nargs": None, "help": ''},
        {"name": "eval_sentences_from", "type": str, "default": 'tacotron/assets/eval_sentences.yaml', "nargs": None, "help": ''},
        {"name": "eval_speakers_from", "type": str, "default": 'tacotron/assets/eval_speakers.yaml', "nargs": None, "help": ''},
        {"name": "eval_unique_spkr", "type": int, "default": 1, "nargs": None, "help": '1 to include unique speaker in evaluation'},
        # data load
        {"name": "batch_size", "type": int, "default": 1, "nargs": None, "help": 'batch size'},
        {"name": "spec_limit", "type": int, "default": 2000, "nargs": None, "help": 'maximum length of spectrogram to include in training set'},
        {"name": "speed_x", "type": float, "default": 1.0, "nargs": None, "help": 'multiply voice speed by this value.'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "speaker_id_list", "type": str, "default": '', "nargs": None, "help": 'speaker id (vctk_p225,son, ...)  to generate speech, seperated by comma'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 0, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "num_recon_iters", "type": int, "default": 50, "nargs": None, "help": '# of iteration in griffin-lim recon'},
        {"name": "spec_pow", "type": float, "default": 1.0, "nargs": None, "help": 'power value for spectrum input in griffin-lim'},
        {"name": "target_type", "type": str, "default": 'auto', "nargs": None, "help": 'output type'},
        {"name": "remove_dummy_pho", "type": int, "default": 1, "nargs": None, "help": '1 to Remove dummy phonemes in g2p'},
        # misc
        {"name": "server", "type": int, "default": 0, "nargs": None, "help": 'run on server (1) or not (0)'},
        {"name": "vocoder_type", "type": str, "default": 'regan', "nargs": None, "help": 'vocoder type: regan/wr/wg'},
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "ignore_version_conflict", "type": int, "default": 0, "nargs": None, "help": 'set 1 to ignore version conflict error'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict


def get_default_bulkconv_args(parse=True):
    args_dict = [
        # path option
        {"name": "gen_idx", "type": int, "default": -1, "nargs": None, "help": 'nth_generation. will be reflected in the generated file name.'},
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "meta_from", "type": str, "default": 'bulk_meta/bulk_meta.txt', "nargs": None, "help": 'load parameters from...'},
        {"name": "out_dir", "type": str, "default": '/home/users/yg/data/bulk_gen_output', "nargs": None, "help": ''},
        # generation option
        {"name": "batch_size", "type": int, "default": 1, "nargs": None, "help": '(batch_size * num_style) wav files will be generated concurrently'},
        {"name": "spec_limit", "type": int, "default": 1000, "nargs": None, "help": 'maximum length of spectrogram to include in training set'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "target_type", "type": str, "default": 'auto', "nargs": None, "help": 'output type: wav/mel/lin'},
        {"name": "lang", "type": str, "default": 'KOR', "nargs": None, "help": 'default language of texa: KOR/ENG'},
        {"name": "convert_eng2kor", "type": int, "default": 1, "nargs": None, "help": 'use english pronunciation converter, set 0 when using mix model.'},
        {"name": "speed_x", "type": float, "default": 1.0, "nargs": None, "help": 'multiply voice speed by this value.'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "seqend_offset", "type": int, "default": 0, "nargs": None, "help": 'Offset to manually adjust expected_seq_end in generation.'},
        {"name": "half", "type": int, "default": 0, "nargs": None, "help": 'set 1 to infer in half precision.'},
        {"name": "no_style", "type": int, "default": 0, "nargs": None, "help": 'set 1 to use only gst_mean for style.'},
        {"name": "adaptive_batch", "type": int, "default": 0, "nargs": None, "help": 'set 1 to adjust batch-size adaptively.'},
        {"name": "num_recon_iters", "type": int, "default": 50, "nargs": None, "help": '# of iteration in griffin-lim recon'},
        # misc
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "n_cpu", "type": int, "default": 1, "nargs": None, "help": 'number of cpu workers, will be ignored in gpu mode'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 0, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "target_lang", "type": str, "default": 'KOR', "nargs": None, "help": 'KOR, ENG, None'},
        {"name": "spec_pow", "type": float, "default": 1.0, "nargs": None, "help": 'power value for spectrum input in griffin-lim'},
        {"name": "ignore_version_conflict", "type": int, "default": 0, "nargs": None, "help": 'set 1 to ignore version conflict error'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict


def get_backward_compatible_args(parse=True):
    args_dict = [
        # data load
        {"name": "data", "type": str, "default": None, "nargs": None, "help": 'dataset to use. comma seperate to use multiple dataset'},
        {"name": "ft_lang", "type": str, "default": '', "nargs": None, "help": 'language specification for fine-tuning'},
        {"name": "ft_gender", "type": str, "default": '', "nargs": None, "help": 'gender specification for fine-tuning'},
        {"name": "config", "type": str, "default": None, "nargs": None, "help": 'path to voxa config'},
        {"name": "use_sg", "type": int, "default": 0, "nargs": None, "help": 'use sg phoneme'},
        {"name": "batch_size", "type": int, "default": 64, "nargs": None, "help": 'batch size'},
        {"name": "text_limit", "type": int, "default": 1000, "nargs": None, "help": 'maximum length of text to include in training set'},
        {"name": "spec_limit", "type": int, "default": 1000, "nargs": None, "help": 'maximum length of spectrogram to include in training set'},
        {"name": "trunc_size", "type": int, "default": 500, "nargs": None, "help": 'used for truncated-BPTT when memory is not enough.'},
        {"name": "shuffle_data", "type": int, "default": 1, "nargs": None, "help": 'whether to shuffle data loader'},
        {"name": "load_queue_size", "type": int, "default": 4, "nargs": None, "help": 'maximum number of batches to load on the memory'},
        {"name": "n_workers", "type": int, "default": 2, "nargs": None, "help": 'number of workers used in data loader'},
        {"name": "no_validation", "type": int, "default": 0, "nargs": None, "help": '1 to use whole data for training'},
        # model
        {"name": "charvec_dim", "type": int, "default": 512, "nargs": None, "help": ''},
        {"name": "enc_hidden", "type": int, "default": 512, "nargs": None, "help": ''},
        {"name": "jpn_acc_dim", "type": int, "default": 32, "nargs": None, "help": ''},
        {"name": "att_hidden", "type": int, "default": 128, "nargs": None, "help": ''},
        {"name": "dec_hidden", "type": int, "default": 256, "nargs": None, "help": ''},
        {"name": "dec_out_size", "type": int, "default": 120, "nargs": None, "help": 'decoder output size'},
        {"name": "post_out_size", "type": int, "default": 513, "nargs": None, "help": 'should be n_fft / 2 + 1(check n_fft from "input_specL" '},
        {"name": "dec_out_type", "type": str, "default": 'mel', "nargs": None, "help": 'output type: lin/mel'},
        {"name": "spkr_embed_size", "type": int, "default": 64, "nargs": None, "help": 'speaker embedding dimension'},
        {"name": "prosody_size", "type": int, "default": 2, "nargs": None, "help": 'prosody embedding dimension'},
        {"name": "key_size", "type": int, "default": 512, "nargs": None, "help": 'dimension of prosody attention key'},
        {"name": "context_hidden_size", "type": int, "default": 512, "nargs": None, "help": 'prosody embedding dimension'},
        {"name": "num_trans_layer", "type": int, "default": 2, "nargs": None, "help": 'number of layers in transformer part'},
        {"name": "r_factor", "type": int, "default": 4, "nargs": None, "help": 'reduction factor(# of multiple output)'},
        {"name": "att_range", "type": int, "default": 9, "nargs": None, "help": 'how many characters to consider when computing attention'},
        {"name": "fluency", "type": int, "default": 0, "nargs": None, "help": '1 to use fluent model, 0 to use non-fluent model'},
        {"name": "dropout", "type": float, "default": 0.5, "nargs": None, "help": ''},
        # optimization
        {"name": "max_epochs", "type": int, "default": 540, "nargs": None, "help": 'maximum epoch to train'},
        {"name": "max_iters", "type": int, "default": -1, "nargs": None, "help": 'maximum iteration to train (approximate)'},
        {"name": "grad_clip", "type": float, "default": 1, "nargs": None, "help": 'gradient clipping'},
        {"name": "learning_rate", "type": float, "default": 1e-3, "nargs": None, "help": '2e-3 from Ito, I used to use 5e-4'},
        {"name": "lr_decay", "type": int, "default": 1, "nargs": None, "help": '1 to use learning rate decay'},
        {"name": "prosody_recon", "type": int, "default": 0.01, "nargs": None, "help": '1 to use learning rate decay'},
        {"name": "teacher_forcing_ratio", "type": float, "default": 1, "nargs": None, "help": 'value between 0~1, use this for scheduled sampling'},
        {"name": "aug_teacher_forcing", "type": int, "default": 1, "nargs": None, "help": '1 to feed generated output as input to make model robust'},
        {"name": "train_durpred", "type": int, "default": 1, "nargs": None, "help": '1 to jointly training a duration predictor'},
        {"name": "spectral_flatness", "type": int, "default": 1, "nargs": None, "help": '1 to consider spectral flatness in loss calculation'},
        {"name": "context_weight", "type": float, "default": 1, "nargs": None, "help": ''},
        {"name": "adv_weight", "type": float, "default": 0.0005, "nargs": None, "help": ''},
        # loading
        {"name": "init_from", "type": str, "default": '', "nargs": None, "help": 'load parameters from...'},
        {"name": "spkr_from", "type": str, "default": '', "nargs": None, "help": 'load speaker embedding layer from...'},
        {"name": "att_extractor_from", "type": str, "default": '', "nargs": None, "help": 'load attention extractor from...'},
        {"name": "aligner_from", "type": str, "default": '', "nargs": None, "help": 'load aligner from...'},
        {"name": "filter_data_file", "type": str, "default": 'loss_summary/filtered_samples_dict.t7', "nargs": None, "help": 'filter out bad examples using...'},
        {"name": "freeze", "type": str, "default": '', "nargs": None, "help": 'freeze params. comma seperate to freeze multiple modules.'},
        {"name": "balanced_spkr", "type": int, "default": 0, "nargs": None, "help": None},
        {"name": "resume", "type": int, "default": 0, "nargs": None, "help": '1 for resume from saved epoch'},
        # misc
        {"name": "exp_no", "type": str, "default": '0', "nargs": None, "help": ''},
        {"name": "pretraining", "type": int, "default": 0, "nargs": None, "help": 'set this to 1 for pretraining model'},
        {"name": "stop_type", "type": int, "default": 0, "nargs": None, "help": 'Choose a logic to stop generation, 0: complex, 1: location of the last att, 2: stop_token'},
        {"name": "print_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "save_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "veripe_every", "type": int, "default": -1, "nargs": None, "help": ''},
        {"name": "save_dir", "type": str, "default": '/nas/yg/data/Tacotron2-seqprosody/checkpoint', "nargs": None, "help": ''},
        {"name": "pinned_memory", "type": int, "default": 1, "nargs": None, "help": '1 to use pinned memory'},
        {"name": "model_version", "type": str, "default": '', "nargs": None, "help": 'used for distinguishing code compatibility'},
        {"name": "mem_prealloc", "type": int, "default": 0, "nargs": None, "help": '1 to use memory preallocation to avoid OOM error'},
        {"name": "gpu", "type": int, "default": [], "nargs": '+', "help": 'index of gpu machines to run'},
        {"name": "eval_set", "type": str, "default": 'default', "nargs": None, "help": 'evaluation set'},
        # debug
        {"name": "debug", "type": int, "default": 0, "nargs": None, "help": '1 for debug mode'},
        {"name": "conv", "type": int, "default": 0, "nargs": None, "help": '1 to use conv encoder.'},
        {"name": "gst", "type": int, "default": 1, "nargs": None, "help": '0-sside;1-gst'},
        {"name": "prosody_type", "type": int, "default": 0, "nargs": None, "help": '0:not_use, 1,2,3: type1,2,3'},
        {"name": "n_token", "type": int, "default": 10, "nargs": None, "help": 'number of style token'},
        {"name": "n_head", "type": int, "default": 0, "nargs": None, "help": 'positive int to use multi-head attention'},
        {"name": "beta", "type": float, "default": 1, "nargs": None, "help": 'beta of beta-VAE'},
        {"name": "prosody_ignore_spkrs", "type": str, "default": None, "nargs": None, "help": 'speakers to ignore prosody prediction (e.g. singing voice)'},
    ]

    if parse:
        return get_args(args_dict)
    else:
        return args_dict


def get_args(args_dict):
    sys_args = sys.argv[1:]
    valid_sys_args_list = sys_args
    args_preset_key_value = {}
    for i, item in enumerate(sys_args):
        if item == '--args_from':
            args_path = sys_args[i + 1]
            with open(args_path, 'r') as args_yaml:
                args_preset_key_value = yaml.load(args_yaml, yaml.SafeLoader)
            print(f"Found a preset file: {args_path}")
            valid_sys_args_list = sys_args[:i] + sys_args[i+2:]
            break

            # merge args from preset with valid_sys_args. (Note that, sys_args take priority over args_preset)
    merged_args = []
    stored_values = []
    key = None
    for item in valid_sys_args_list:
        if item.startswith('--'):
            if key is not None:
                merged_args.extend([f'--{key}', *stored_values])
            key = item[2:]  # drop preceding '--'
            stored_values = []
        else:
            stored_values.append(item)
    if len(stored_values) > 0 and key is not None:
        merged_args.extend([f'--{key}', *stored_values])

    for k, v in args_preset_key_value.items():
        if f'--{k}' not in merged_args:
            merged_args.extend([f'--{k}', v])

    # set parser
    parser = argparse.ArgumentParser(description='training script')
    for arg in args_dict:
        if arg['nargs'] is None:
            parser.add_argument(f'--{arg["name"]}', type=arg['type'], default=arg['default'], help=arg['help'])
        else:
            parser.add_argument(f'--{arg["name"]}', type=arg['type'], default=arg['default'], nargs=arg['nargs'], help=arg['help'])
    return parser.parse_args(args=merged_args)


_exclude_override_args = ('exp_no',)
def load_and_override_args(saved_args, new_args):
    if not type(new_args) is dict:
        new_args = vars(new_args)

    # load and override some arguments
    for key in new_args.keys():
        if key in _exclude_override_args:
            continue
        saved_args.__dict__[key] = new_args[key]

    # make it compatible with current args
    default_args = get_backward_compatible_args(parse=False)
    for d in default_args:
        key = d['name']
        if not key in saved_args.__dict__.keys():
            saved_args.__dict__[key] = d['default']

    return saved_args

