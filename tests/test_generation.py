from tts_text_util.process_text_input import TextToInputSequence
from tacotron.synthesize import load_model, infer_one
from tacotron.util.default_args import get_default_synthesis_args

MODEL_PATH_TYPECAST1909='/nas/shared/steach_checkpoint/tacotron/356_242_rearranged.t7'
MODEL_PATH_AUDIOBOOK1912='/nas/shared/steach_checkpoint/tacotron/471_240.t7'
MODEL_PATH_OLD_LINEAR_MODEL='/nas/shared/steach_checkpoint/tacotron/200420-typecast_ko_f_2004-090e39f_240sty.t7'
MODEL_PATH_ALIGN2105_MODEL='/nas/shared/steach_checkpoint/tacotron/210531-jidam_rapper-c944a33_laststy.t7'
MODEL_PATH_CATS1_MODEL='/nas/shared/steach_checkpoint/tacotron/210715-cats_mixm_60.t7'
MODEL_PATH_CATS2_MODEL='/nas/shared/steach_checkpoint/tacotron/210802_cats_tobi_laststy.t7'
MODEL_PATH_CATS3_MODEL='/nas/shared/steach_checkpoint/tacotron/210115_cats3_test.t7'
MODEL_PATH_CATS4_MODEL='/nas/shared/steach_checkpoint/tacotron/220711_cats4_mixf_base.t7'
MODEL_PATH_LATEST_MODEL=MODEL_PATH_CATS4_MODEL
SPEAKER_ID='etri_F'
TEST_OUT_DIR='/nas/shared/co-work/tts_test_output'
TEST_INPUT_KOR="위의 예에서 어떤 대우 표현을 선택하여 쓰느냐 하는 것은 말할이의 교양이나 품위와 밀접한 관련이 있다."

def test_synthesize_typecaset1909():
    synthesize_common(MODEL_PATH_TYPECAST1909, remove_dummy_pho=False)
    synthesize_style_speed(MODEL_PATH_TYPECAST1909, remove_dummy_pho=False)


def test_synthesize_audiobook1912():
    synthesize_common(MODEL_PATH_AUDIOBOOK1912, remove_dummy_pho=False)


def test_synthesize_old_linear_model():
    synthesize_common(MODEL_PATH_OLD_LINEAR_MODEL, remove_dummy_pho=False)
    synthesize_style_speed(MODEL_PATH_OLD_LINEAR_MODEL, remove_dummy_pho=False)


def test_synthesize_align2105_model():
    synthesize_common(MODEL_PATH_ALIGN2105_MODEL, 'jidam_rapper')


def test_synthesize_catsv1_model():
    synthesize_common(MODEL_PATH_CATS1_MODEL, 'etri_M')


def test_synthesize_catsv2_model():
    synthesize_common(MODEL_PATH_CATS2_MODEL, 'suhyun_tobi_emo_angry')

def test_synthesize_catsv3_model():
    synthesize_common(MODEL_PATH_CATS3_MODEL, 'etri_F')

def test_synthesize_latest_model():
    synthesize_common(MODEL_PATH_LATEST_MODEL, 'etri_F')


def synthesize_common(model_path, speaker_id=None, remove_dummy_pho=True):
    new_args = get_default_synthesis_args()
    new_args.init_from = model_path
    if speaker_id:
        new_args.tgt_spkr = speaker_id
    else:
        new_args.tgt_spkr = SPEAKER_ID
    new_args.out_dir = TEST_OUT_DIR

    loaded = load_model(new_args)

    phoneme_util = TextToInputSequence(True, use_sg=loaded.args.use_sg)
    speaker_lang = new_args.target_lang      # This should be replaced by the language of the forced speaker.
    text_processed = phoneme_util.prepare_input(TEST_INPUT_KOR, new_args.target_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
    infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
              loaded.speaker_manager, loaded.data_list, None, True)

def synthesize_style_speed(model_path, speaker_id=None, remove_dummy_pho=True):
    new_args = get_default_synthesis_args()
    new_args.init_from = model_path
    if speaker_id:
        new_args.tgt_spkr = speaker_id
    else:
        new_args.tgt_spkr = SPEAKER_ID
    new_args.out_dir = TEST_OUT_DIR
    
    new_args.gst_source = 'cluster'
    new_args.gst_idx = 0
    new_args.speed = 0.22

    loaded = load_model(new_args)

    phoneme_util = TextToInputSequence(True, use_sg=loaded.args.use_sg)
    speaker_lang = new_args.target_lang      # This should be replaced by the language of the forced speaker.
    text_processed = phoneme_util.prepare_input(TEST_INPUT_KOR, new_args.target_lang, speaker_lang, rm_dummy_ph=remove_dummy_pho)
    infer_one(loaded.model, text_processed, loaded.args, loaded.style_list, loaded.voxa_config,
              loaded.speaker_manager, loaded.data_list, None, True)
              

def test_evaluation_linear_model():
    pass


def test_evaluation_mel_model():
    pass


def test_bulk_gst_linear_model():
    pass


def test_bulk_gst_mel_model():
    pass
