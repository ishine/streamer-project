from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline, set_seed
### from models.py
# from models import *
import torch
import random
import time

# import sounddevice as sd

import os, sys, multiprocessing
from scipy.io import wavfile
import tts_text_util as texa
from tts_text_util.process_text_input import TextToInputSequence

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)

from tacotron.synthesize import load_model, infer_one
from tacotron.util.default_args import get_default_synthesis_args

badwords = """
arse
ass
asshole
bastard
bitch
bollocks
brotherfucker
bugger
bullshit
child-fucker
Christ on a bike
Christ on a cracker
cocksucker
crap
cunt
damn
fatherfucker
frigger
fuck
fucker
goddamn
godsdamn
hell
holy shit
horseshit
in shit
Jesus Christ
Jesus fuck
Jesus H. Christ
Jesus Harold Christ
Jesus wept
Jesus, Mary and Joseph
motherfucker
nigga
piss
prick
shit
shit ass
sisterfucker
slut
son of a bitch
son of a whore
sweet Jesus
tit
twat
"""

badwords = badwords.split('\n')[1:-1]

if __name__ == '__main__':

    new_args = get_default_synthesis_args()

    bj = pipeline('text-generation', model='gpt2-large')
    set_seed(42)


    mname = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForSequenceClassification.from_pretrained(mname).cuda(7)

    chatbot_mname = "facebook/blenderbot-3B"
    chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(chatbot_mname)
    chatbot = BlenderbotForConditionalGeneration.from_pretrained(chatbot_mname).cuda(7)

    loaded = load_model(new_args)
    phoneme_util = TextToInputSequence(True, use_sg=loaded.args.use_sg, pls=False)
    speaker_lang = 'ENG'      # This should be replaced by the language of the forced speaker.
    remove_dummy_pho = True if new_args.remove_dummy_pho == 1 else False

    label = ['HateSpeech','Normal','Offensive']

    # model = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two").cuda(7)
    os.system('clear')

    start = input('start? press ENTER')

    prompt = input('What topic are you interested in? ')

    generated = bj(prompt, max_length=32, num_return_sequences=1)[0]['generated_text']

    text_processed = phoneme_util.prepare_input(generated, speaker_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
    output = infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
                loaded.speaker_manager, loaded.data_list, None, True, True)

    wavfile.write('storytelling.wav',44100,output)
    # sd.play(output,44100)

    print(f"ILoveTotoro: {generated}")

    stop = input('start? press ENTER')

    chats = [
        'Swordmaster : I hate this story. Stop mother fucker.',
        'Naruto : I love this story!',
        'Kyojuro : I should go hospital tonight.. Do you know why?',
        'Luffy : Do you know that? this bastard is a nazi!',
        'Hater : Have you ever been to Paris? What a awesome place!',
        'Noel : What the Fucking world! I hate this world! I wanna kill you all, bitch!',
        'Younggun : I will have lunch seperately.',
        "Changu : I'm so hungry. Recommend a menu!",
        'Taesu : The future of creativity driven by Artificial Intelligence!'
    ]

    print()
    print('----------------------------------------------------------------')
    print()

    for c in chats :
        time.sleep(1)
        print(c)
        
    time.sleep(1)
    print()
    print('----------------------------------------------------------------')
    print()

    inputs = [ch.split(' : ')[1] for ch in chats]
    inputs = tokenizer(inputs, return_tensors="pt",padding=True)
    item = model(input_ids=inputs['input_ids'].cuda(7),attention_mask=inputs['attention_mask'].cuda(7))

    pred = torch.argmax(item.logits.detach().cpu(),dim=1).tolist()

    to_respond = []

    hate = False

    print('--Reply candidate--')
    print()

    for ch, p in zip(chats, pred) :
        if label[p] == 'Normal':
            for bad in badwords :
                if bad.lower() in ch.lower() :
                    hate = True
                    break
            if not hate :
                to_respond.append(ch)
                print(f'{ch} - {label[p]}')
                time.sleep(1)
            hate = False
                    
    print()
    print('----------------------------------------------------------------')
    print()

    selected = random.choice(to_respond)

    UTTERANCE = selected + '</s> <s>'
    time.sleep(1)
    print(f'selected chat : {selected}')
    print()

    inputs = chatbot_tokenizer([UTTERANCE], return_tensors="pt")
    input_ids = inputs.input_ids.cuda(7)
    reply_ids = chatbot.generate(inputs=input_ids)
    bot_reply = chatbot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    text_processed = phoneme_util.prepare_input(f"{UTTERANCE.split(' : ')[0]},{bot_reply}", speaker_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
    output = infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
            loaded.speaker_manager, loaded.data_list, None, True, True)

    wavfile.write('reply.wav',44100,output)
    # sd.play(output,44100)
    print(f"ILoveTotoro: {UTTERANCE.split(' : ')[0]},{bot_reply}")