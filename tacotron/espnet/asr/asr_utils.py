#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os

import torch

from tacotron.espnet.nets import E2E
from tacotron.espnet.nets import Loss


# matplotlib related
import matplotlib
matplotlib.use('Agg')


# * -------------------- general -------------------- *
class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def __getitem__(self, name):
        return self.obj[name]

    def __len__(self):
        return len(self.obj)

    def fields(self):
        return self.obj

    def items(self):
        return self.obj.items()

    def keys(self):
        return self.obj.keys()


def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json)

    :param str model_path: model path
    :param str conf_path: optional model config path
    """

    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        return json.load(f, object_hook=AttributeDict)


def torch_save(path, model):
    """Function to save torch model states

    :param str path: file path to be saved
    :param torch.nn.Module model: torch model
    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Function to load torch model states

    :param str path: model file or snapshot file to be loaded
    :param torch.nn.Module model: torch model
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def load_espnet():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int, help='Number of GPUs')
    parser.add_argument('--backend', default='chainer', type=str, choices=['chainer', 'pytorch'], help='Backend library')
    parser.add_argument('--debugmode', default=1, type=int, help='Debugmode')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--verbose', '-V', default=1, type=int, help='Verbose option')
    parser.add_argument('--batchsize', default=1, type=int, help='Batch size for beam search (0: means no batch processing)')
    # task related
    # parser.add_argument('--recog-json', type=str,
    #                     help='Filename of recognition data (json)')
    parser.add_argument('--result-label', type=str, required=True, help='Filename of result label data (json)')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True, help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None, help='Model config file')
    # search related
    parser.add_argument('--nbest', type=int, default=1, help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1, help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float, help='Incertion penalty')
    parser.add_argument('--maxlenratio', default=0.0, type=float, help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', default=0.0, type=float, help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', default=0.0, type=float, help='CTC weight in joint decoding')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None, help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None, help='RNNLM model config file to read')
    parser.add_argument('--word-rnnlm', type=str, default=None, help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None, help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None, help='Word list to read')
    parser.add_argument('--lm-weight', default=0.1, type=float, help='RNNLM weight.')

    # Arguments for pre-trained model
    args_example = "--ngpu 0 --backend pytorch --batchsize 0"
    args_example += " --result-label temp/data.json"
    args_example += " --model /nas/yg/project/librispeech_ns_phone/exp/train_960_pytorch_vggblstm_e5_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs20_mli800_mlo150/results/model.acc.best"
    args_example += " --beam-size 20 --penalty 0.0 --maxlenratio 0.0 --minlenratio 0.0 --ctc-weight 0.5 --lm-weight 0"
    esp_args = parser.parse_args(args_example.split())

    # read training config
    idim, odim, train_args = get_model_conf(esp_args.model, esp_args.model_conf)
    model = Loss(E2E(idim, odim, train_args), 1)
    torch_load(esp_args.model, model)

    return model, esp_args, train_args


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js
