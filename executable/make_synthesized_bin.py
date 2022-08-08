# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os.path, yaml, sys
import os, argparse, torch, pickle
from tqdm import tqdm
from voxa import VoxaConfig
from voxa.database.data_group import expand_data_group
from voxa.speakermgr.speaker_manager import SpeakerManager
from voxa.prep.signal_processing import SignalProcessing

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.util.default_args import get_default_synthesized_bin_args
from tacotron.util.gen_common import auto_clamp


def get_metadata(args, metafile, whitelist=None, blacklist=None):
    """ read metadata and make a list of files
        And write text file while preserving order.
        whitelist (set): ONLY use files in this list
        blacklist (set): use files EXCEPT in this list
        preprocessing (bool): Don't write text file. (If you call this at the outside of this code, use False.)
    """
    assert whitelist is None or blacklist is None

    metalist = []
    rFile = open(args.dir_home + metafile, 'r')

    line = rFile.readline()
    while line:
        filename, text, lang = line.strip().split('|')

        # filter files if needed.
        if whitelist is not None and filename not in whitelist:
            continue
        if blacklist is not None and filename in blacklist:
            continue

        metalist.append((filename, text, lang))
        line = rFile.readline()

    rFile.close()
    return metalist

def reprocess_audio(args, old_line_load_dict, mel_path, txt_path, spkr_id, model, taco_args):
    bin_file_dict = {}
    target_list_dict = {}
    serialized_dict = {}
    offset_dict = {}

    # TODO: move bin_path_dict to voxa
    bin_path_dict = {
        "output_post": voxa_config.synth_mel_binary,
        "output_dec": None,
        "taco_lstm_out": voxa_config.taco_dec_out_binary,
        "att_context": voxa_config.att_ctx_binary,
        "phoneme_duration": 'phoneme_duration_binary.bin',
        # "phoneme_duration": voxa_config.phoneme_duration_binary,
    }
    sp = SignalProcessing(voxa_config)

    for feature_name in model.variables_to_keep:
        bin_file_dict[feature_name] = open(os.path.join(args.dir_bin, bin_path_dict[feature_name]), 'wb')
        target_list_dict[feature_name] = []
        offset_dict[feature_name] = 0

    for load_dict in tqdm(old_line_load_dict['data']):
        # If out of range error occurs at here, check whether you are using right text_limit.
        offset_mel, read_n_mel = load_dict['offset_mel'], load_dict['read_n_mel']
        offset_phone, read_n_phone = load_dict['offset_phone'], load_dict['read_n_phone']

        target_mel_org = torch.from_numpy(load_binary(mel_path, offset_mel, read_n_mel)).float().unsqueeze(0)
        target_mel_pad = torch.Tensor(target_mel_org.size(0), taco_args.r_factor, target_mel_org.size(2)).zero_()
        phoneme_input = torch.LongTensor(load_binary(txt_path, offset_phone, read_n_phone)).unsqueeze(0)
        spkr_id = torch.LongTensor([spkr_id])


        spec_length_original = target_mel_org.size(1)
        phoneme_lengths = phoneme_input.size(1)
        target_mel = torch.cat((target_mel_org, target_mel_pad), dim=1)
        subbatch_spec_lengths = torch.Tensor([spec_length_original+taco_args.r_factor]).div(taco_args.r_factor).ceil().mul(
            taco_args.r_factor).int().add_(-taco_args.r_factor).tolist()  # consider r_factor
        spec_lengths = torch.Tensor(subbatch_spec_lengths)
        phoneme_lengths = torch.IntTensor([phoneme_lengths]).add(1).tolist()

        # padding
        target_mel_new = torch.Tensor(target_mel.size(0), subbatch_spec_lengths[0], target_mel.size(2)).zero_()
        phoneme_input_new = torch.LongTensor(phoneme_input.size(0), phoneme_lengths[0]).zero_()
        phoneme_input_new[:, 0:phoneme_input.size(1)].copy_(phoneme_input)
        phoneme_input = phoneme_input_new

        if args.gpu:
            phoneme_input = phoneme_input.cuda()
            spkr_id = spkr_id.cuda()
            spec_lengths = spec_lengths.cuda()

        # synthesize mel with teacher forcing
        output_post_dict = {}
        target_mel_new.zero_()
        target_mel_new[:, 0:spec_length_original].copy_(target_mel[:,0:spec_length_original])
        target_mel = target_mel_new
        if args.gpu:
            target_mel = target_mel.cuda()

        model.reset_decoder_states(debug=taco_args.debug)
        model_out_dict = model(phoneme_input, target_mel, spkr_id, subbatch_spec_lengths, phoneme_lengths,
                               whole_spec_len=spec_lengths, debug=taco_args.debug, target_mel_whole=target_mel)
        # check the output post is whether mel or linear
        output_post = model_out_dict.get('output_post')[0, :spec_length_original]
        output_post = auto_clamp(output_post, sp, taco_args.dec_out_type).cpu().numpy()
        if taco_args.dec_out_type == 'lin':
            output_post = sp.lin2mel(output_post, spow=1.0, is_numpy=True)

        # output interpolation step
        # mel_gta_org = output_dec_dict[0].clone()
        model_out_dict['output_post']=output_post

        for feature_name in model.variables_to_keep:
            feature = model_out_dict.get(feature_name)

            if feature_name == 'output_dec':
                pass
            elif feature_name == 'output_post':
                serialized_dict[feature_name] = pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL)
            elif feature_name == 'taco_lstm_out':
                serialized_dict[feature_name] = pickle.dumps(feature[0].cpu().numpy(), protocol=pickle.HIGHEST_PROTOCOL)
            elif feature_name == 'att_context':
                serialized_dict[feature_name] = pickle.dumps(feature[0].cpu().numpy(), protocol=pickle.HIGHEST_PROTOCOL)
            elif feature_name == 'phoneme_duration':
                serialized_dict[feature_name] = pickle.dumps(feature[0].cpu().numpy(), protocol=pickle.HIGHEST_PROTOCOL)

            # write to file
            bin_file_dict[feature_name].write(serialized_dict[feature_name])

            # generate audio information list
            target_list_dict[feature_name].append([offset_dict[feature_name], len(serialized_dict[feature_name])])
            offset_dict[feature_name] += len(serialized_dict[feature_name])

    for feature_name in model.variables_to_keep:
        bin_file_dict[feature_name].close()

    return target_list_dict


def reprocess_dataset(data, model, speaker_manager, voxa_config):
    is_dataset_exist = False
    with open(voxa_config.dataset_list_file, 'r') as rFile:
        line = rFile.readline()
        while line:
            if line.split("|")[0] == data:
                _, args.dir_home, dir_bin = line.strip().split('|')
                args.dir_bin = os.path.join(voxa_config.hdd_bin_root, dir_bin, voxa_config.alias)
                is_dataset_exist = True
                break
            line = rFile.readline()

    if not is_dataset_exist:
        print('No such data:', data)
        return

    if args.use_sg == 1:
        txt_path = os.path.join(args.dir_bin, voxa_config.phoneme_sg_binary)
    else:
        txt_path = os.path.join(args.dir_bin, voxa_config.phoneme_binary)
    load_dict_filename = os.path.join(args.dir_bin, voxa_config.load_dict_file)
    load_dict_filename = os.path.join(args.dir_bin, voxa_config.load_dict_file)
    mel_path = os.path.join(args.dir_bin, voxa_config.mel_spec_binary)
    spkr_id = speaker_manager.get_compact_id(data)

    line_load_dict = torch.load(load_dict_filename)

    # Start reprocessing (Binarize tacotron output)
    with torch.no_grad():
        reprocessed_dict = reprocess_audio(args, line_load_dict, mel_path, txt_path, spkr_id, model, taco_args)

    # make file load dict
    for i, d in enumerate(line_load_dict['data']):
        for feature in model.variables_to_keep:
            line_load_dict['data'][i][f'offset_{feature}'] = reprocessed_dict[feature][i][0]
            line_load_dict['data'][i][f'read_n_{feature}'] = reprocessed_dict[feature][i][1]
            #d[f'offset_{feature}'] = reprocessed_dict[feature][i][0]
            #d[f'read_n_{feature}'] = reprocessed_dict[feature][i][1]

    torch.save(line_load_dict, load_dict_filename)


def load_binary(file_path, offset, length):
    with open(file_path, 'rb') as datafile:
        datafile.seek(offset)
        line = datafile.read(length)
        obj = pickle.loads(line)
    return obj


if __name__ == '__main__':
    valid_feats = set(["output_dec","output_post", "taco_lstm_out", "att_context", "phoneme_duration"])
    args = get_default_synthesized_bin_args()

    target_feats = args.feature.split(',')
    isValid = True
    for f in target_feats:
        if f not in valid_feats:
            isValid = False
            break

    if len(target_feats) == 0 or not isValid:
        print("Must specify feature: output_dec, taco_lstm_out, att_context, phoneme_duration")
        raise RuntimeError()

    # trim silence before preprocessing.
    print('Dataset to reprocess:', args.data)

    # set global whitelist or global blacklist
    g_whitelist = None
    g_blacklist = None
    if args.global_white_list == 1 and os.path.isfile('/home/data2/database/global.whitelist'):
        g_whitelist = set([])
        with open('/home/data2/database/global.whitelist', 'r') as rFile:
            line = rFile.readline()
            while line:
                g_whitelist.add(line.strip())
                line = rFile.readline()

    if args.global_black_list == 1 and os.path.isfile('/home/data2/database/global.blacklist'):
        g_blacklist = set([])
        with open('/home/data2/database/global.blacklist', 'r') as rFile:
            line = rFile.readline()
            while line:
                g_blacklist.add(line.strip())
                line = rFile.readline()

    # load Tacotron
    checkpoint = torch.load(args.taco_from, map_location=lambda storage, loc: storage)
    taco_args = checkpoint['args']

    # set voxa config
    if args.config is None:
        voxa_config = VoxaConfig.load_from_config(checkpoint['config'])
    else:
        voxa_config = VoxaConfig(args.config)
    # convert data_group into list of single speaker data list
    converted_data_list = expand_data_group(args.data.split(','), voxa_config, sort=False)

    assert taco_args.use_sg == args.use_sg
    assert taco_args.gst == 0 or taco_args.gst == 1
    if taco_args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron

    model = Tacotron(taco_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.reset_decoder_states()

    print('loaded checkpoint %s' % (args.taco_from))

    data_list = sorted(expand_data_group(taco_args.data.split(','), voxa_config))
    speaker_manager = SpeakerManager(data_list)

    if args.gpu is None:
        args.gpu = []
    else:
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model = model.cuda()

    model = model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
        elif isinstance(m, torch.nn.Dropout):
            m.eval()

    for feature in target_feats:
        model.keep_features(feature)

    for i, data in enumerate(converted_data_list):
        # No need for multi-processing since STFT already utilizes multi-processing
        print(f'Processing: {data}, ({i+1}/{len(converted_data_list)})')
        reprocess_dataset(data, model, speaker_manager, voxa_config)
