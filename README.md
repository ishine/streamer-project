# Tacotron2-seqprosody

## Prerequisite
* python3.6
* torch
* tts_text_util
* voxa
* hgtk

## Training
It is recommended to use [Steach](https://github.com/neosapience/steach).

For manual training,
```bash
# Copy binary files for training. (Make sure you have enough space in ssd.)
voxa rsync \
    --data {datasets-to-use} \      # comma separated list of dataset (ex. 'etri_F,etri_M', check /nas/shared/dataset_info/dataset_list.txt)
    --waveform 0 \                  # Skip copying binary file of waveform
    --waveform_h 0                  # Skip copying binary file of high-freq waveform 
    --waveform_h2 0                 # Skip copying binary file of high-freq waveform 
    --spec_lin 0                    # Skip copying binary file of linear spectrogram

# Start training.
python executable/train.py  \
    --data {datasets-to-use}  \     # comma separated list of dataset (ex. 'etri_F,etri_M')
    --gpu {gpu-id-to-use}  \        # gpu index to use
    --exp_no {experiment-index}     # experiment ID for logging and checkpoint naming
```
For more arguments, check "default_args.py"

### You can set arguments using an yaml file instead of command line input.
```bash
python executable/train.py --args_from {path-to-preset-yaml}       # Check the example: tacotron/assets/args_preset_sample.yaml
```

## Generation

### generate one sentence with average style of a speaker
```bash
python executable/synthesize.py \
    --init_from {path-to-checkpoint} \
    --tgt_spkr {target-speaker-id} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} \
    --caption {sentence-to-generate}
```

### generate one sentence with style (style from saved style index)
```bash
python executable/synthesize.py \
    --gst_source cluster \
    --gst_idx {gst-index} \
    --init_from {path-to-checkpoint} \
    --ref_spkr {source-speaker-id} \
    --tgt_spkr {target-speaker-id} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} \
    --caption {sentence-to-generate}
```

### generate one sentence with transferred style from a reference audio file.
```bash
python executable/synthesize.py \
    --gst_source ref_wav \
    --ref_wav {path-to-reference-audiofile} \
    --init_from {path-to-checkpoint} \
    --ref_spkr {source-speaker-id} \
    --tgt_spkr {target-speaker-id} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} \
    --caption {sentence-to-generate}
```

### generate one sentence with transferred style from a prosody vector file.
```bash
python executable/synthesize.py \
    --gst_source prosody_vector \
    --prosody_ref_file {path-to-prosody-vector-file} \
    --init_from {path-to-checkpoint} \
    --ref_spkr {source-speaker-id} \
    --tgt_spkr {target-speaker-id} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} \
    --caption {sentence-to-generate}
```

### generate "multiple" sentences with average style of a speaker
```bash
python executable/synthesize.py \
    --init_from {path-to-checkpoint} \
    --tgt_spkr {target-speaker-id} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} \
    --ref_txt {path-to-file-with-sentences-to-generate}
```

### generate "multiple" sentences to morph "multiple" speakers
```bash
python executable/synthesize.py     \
    --init_from {path-to-checkpoint}       \
    --tgt_spkr {dummy-speaker-id}  \
    --morph_spkrs {target-speaker-id1, target-speaker-id2, ....}  \
    --gpu 0 \
    --morph_ratio {ratio-for-speaker-id1, ratio-for-speaker-id2, ...}    \
    --ref_txt {path-to-file-with-sentences-to-generate} 
```

### generate audiobook with a metafile
```bash
python executable/bulk_gst.py \
    --init_from {path-to-checkpoint} \
    --meta_from {path-to-metafile} \
    --gen_idx {generation-index} \
    --gpu {gpu-id-to-use} \
    --out_dir {output-path} 
    # --convert_eng2kor 0       # uncomment this to use mix-language model.
```
For detailed examples, check /nas/shared/audiobook/

## Misc

### embed style clusters in the trained model
```bash
python executable/style_cluster.py \
    --init_from {path-to-checkpoint} \
    --target_spkrs {speaker-ids-to-cluster-styles} \    # comma separated (ex. 'etri_F,etri_M')
    --num_clusters {number-of-style-clusters} \         # N + 1 style will be generated (index 0 is reserved for avg style)
    --gpu {gpu-id-to-use}
```
The style-clustered checkpoint will be saved as {exp_no}_{epoch}sty.t7.  
You can specify the new checkpoint path with "--out_path" option.  
If --target_spkrs is not specified, the whole speakers will be clustered. (takes ~1 day.)

### Generate binary file of Tacotron-synthesized mel.
```bash
python executable/make_synthesized_bin.py \
    --data {datasets-to-use}  \     # comma separated (ex. 'etri_F,etri_M')
    --taco_from {path-to-checkpoint} \
    --gpu {gpu-id-to-use}
```

### Test code before sending pull request
```bash
# install pytest if not exists
pip3 install pytest

# test
python3 -m pytest -lvs

# listen to the generated samples
cd /nas/shared/co-work/tts_test_output
```

