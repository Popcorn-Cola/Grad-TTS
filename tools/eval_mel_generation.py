import argparse
import json

import torch
import params
import numpy as np
from model import GradTTS
from data import TextMelDataset
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys, os, re

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max
n_feats = params.n_feats

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def find_files(root_dir, prefix='grad_', suffix='.pt'):
    checkpoint_list = []
    ckpts_id = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith(prefix) and file.endswith(suffix):
                checkpoint_list.append(os.path.join(root, file))
                pattern = re.compile(r'grad_(\d+)\.pt')
                match = pattern.search(file)
                Epoch_ind = int(match.group(1))
                ckpts_id.append(Epoch_ind)
    return checkpoint_list, ckpts_id

def save_mel_spectrograms_to_file(mel_spectrograms, output_file): # input a list of mel tensors
    with open(output_file, 'w') as f:
        for mel_spec in mel_spectrograms:
            # Convert Mel spectrogram to string and write to file
            f.write(','.join(map(str, mel_spec.cpu().numpy())) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str, required=True,
                        help='path to checkpoint directory of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None,
                        help='speaker id for multispeaker model')
    args = parser.parse_args()

    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    #get cmu dictionary
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    # import cmudict
    print('Logging test batch...')
    test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('build test batch...')
    idx = np.random.choice(list(range(len(test_dataset))), size=params.test_size, replace=False)
    test_batch_text = []
    test_batch_mel = []
    for index in idx:
        filepath_and_text = test_dataset.filepaths_and_text[index]
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        mel = test_dataset.get_mel(filepath)

        test_batch_text.append(text)                          #[{'y': mel, 'x': text}, {'y': mel, 'x': text}, {'y': mel, 'x': text}]
        test_batch_mel.append(mel)

    print('output original mel spectrogram and text')
    print('all mel spectrogram is written in a file')
    print('all text is written in a file')
    if not os.path.exists('eval/converted'):
        os.makedirs('eval/converted')
        os.makedirs('eval/original')

    with open('eval/original/text.txt', 'w') as text_file:
        for i, item in enumerate(test_batch_text):
            text_file.write(f"{test_batch_text[i]}\n")

    with open('eval/original/text.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    save_mel_spectrograms_to_file(test_batch_mel, 'eval/original/mel.txt')

    print('get all checkpts file index')
    checkpoint_list, ckpts_id = find_files(args.checkpoint_dir)

    print('Initialize GradTTS MODEL')
    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    for index, checkpoint in enumerate(checkpoint_list):
        y_mel = []
        generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
        _ = generator.cuda().eval()
        print(f'Doing the {index}st epoch')
        os.makedirs(f'eval/converted/Epoch_{ckpts_id[index]}')
        with torch.no_grad():
           for i, text in enumerate(texts):
                    # convert word to phonemes:
                    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                           stoc=False, spk=spk, length_scale=0.91)
                    y_mel.append(y_dec)

        save_mel_spectrograms_to_file(y_mel, f'eval/converted/Epoch_{ckpts_id[index]}/Epoch_{ckpts_id[index]}_mel.txt')




