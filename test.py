from __future__ import unicode_literals, print_function, division

# from playsound import playsound
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import sounddevice as sd

import os, sys, multiprocessing
from scipy.io import wavfile
import tts_text_util as texa
from tts_text_util.process_text_input import TextToInputSequence

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)

from tacotron.synthesize import load_model, infer_one
from tacotron.util.default_args import get_default_synthesis_args

if __name__ == '__main__':
    new_args = get_default_synthesis_args()
     
    # mname = "facebook/blenderbot-400M-distill"
    # mname = "hyunwoongko/blenderbot-9B"
    mname = "facebook/blenderbot-3B"

    # loaded = load_model(new_args)
    # phoneme_util = TextToInputSequence(True, use_sg=loaded.args.use_sg, pls=False)
    # speaker_lang = 'ENG'      # This should be replaced by the language of the forced speaker.
    # remove_dummy_pho = True if new_args.remove_dummy_pho == 1 else False

    model = BlenderbotForConditionalGeneration.from_pretrained(mname).cuda(1)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    ALL_UTTERANCE = ''
    patience = input('How many times will you talk about one topic with Bot? ',)
    stack = 0

    fname = 'reply.wav'

    while True :

        if stack == patience :
            ALL_UTTERANCE = ''
            stack = 0

        UTTERANCE = input("to Bot : ")

        if 'bye' in UTTERANCE.lower() :
            break

        ALL_UTTERANCE += UTTERANCE
        stack += 1
        ALL_UTTERANCE += '</s> <s>'

        inputs = tokenizer([ALL_UTTERANCE], return_tensors="pt")
        input_ids = inputs.input_ids.cuda(1)
        reply_ids = model.generate(inputs=input_ids)
        bot_reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        # text_processed = phoneme_util.prepare_input(bot_reply, speaker_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
        # output = infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
        #           loaded.speaker_manager, loaded.data_list, None, True, True)
        # wavfile.write(fname,16000,output)
        # playsound(fname)
        # sd.play(output,16000)
        print("Bot: ", bot_reply)
        ALL_UTTERANCE += bot_reply + '</s> <s>'
