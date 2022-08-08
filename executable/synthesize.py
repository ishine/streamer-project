# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, sys, multiprocessing

import tts_text_util as texa
from tts_text_util.process_text_input import TextToInputSequence

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.synthesize import load_model, infer_one
from tacotron.util.default_args import get_default_synthesis_args

def main(new_args):
    assert (new_args.ref_txt != '') + (new_args.caption != '') == 1
    if new_args.caption:
        # generate speech for one sentence
        text_inputs = [new_args.caption]
    else:
        # generate speech for multiple sentences in the file 'ref_txt'
        with open(new_args.ref_txt, 'r') as rFile:
            text_inputs = []
            line = rFile.readline()
            while line:
                text_inputs.append(line.strip())
                line = rFile.readline()

    loaded = load_model(new_args)
    phoneme_util = TextToInputSequence(True, use_sg=loaded.args.use_sg)
    speaker_lang = new_args.target_lang      # This should be replaced by the language of the forced speaker.
    remove_dummy_pho = True if new_args.remove_dummy_pho == 1 else False
    for j, text_input in enumerate(text_inputs):
        text_processed = phoneme_util.prepare_input(text_input, new_args.target_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
        infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
                  loaded.speaker_manager, loaded.data_list, loaded.lang_dict, None, True, True)


if __name__ == '__main__':
    try:
        new_args = get_default_synthesis_args()
        main(new_args)
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
