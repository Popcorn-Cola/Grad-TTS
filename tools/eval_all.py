import argparse
import json

import torch
import params
import shutil
import numpy as np
from model import GradTTS
from data import TextMelDataset
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from scipy.io.wavfile import write

import matplotlib.pyplot as plot
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

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]

def save_mel_spectrograms_to_file(mel_spectrograms, output_dir): # input a list of mel tensors, output one mel tensor to one file in output_dir
    for i, mel_spec in enumerate(mel_spectrograms):

        torch.save(mel_spec , f'{output_dir}/mel_{i}.pt')

def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    spec = pt
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    subfig = fig.add_subplot()
    image = subfig.imshow(spec, cmap="jet", origin="lower", aspect="equal", interpolation="none", vmax=vmax,
                          vmin=vmin)
    fig.colorbar(mappable=image, orientation='vertical', ax=subfig, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()

def get_integer_part(s):
    return int(''.join(filter(str.isdigit, s)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', type=str, required=True,
                        help='path to checkpoint directory of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    parser.add_argument('-g', '--gt_dir', type=str, required=False, default='eval/original',
                        help='location to save the ground truth data')
    parser.add_argument('-z', '--cvt_dir', type=str, required=False, default='eval/converted',
                        help='location to save the converted data')
    parser.add_argument('-i', '--epoch_interval', type=int, required=False, default=100,
                        help='The interval between epochs to be evaluated')
    parser.add_argument('-f', '--file', type=str, required=False, default='eval/original/text.txt',
                        help='location of file to contain validation text')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    cvt_dir = args.cvt_dir
    checkpoint_dir = args.checkpoint_dir
    epoch_interval = args.epoch_interval
    # get cmu dictionary
    # importcmudict
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    # import cmudict
    print('Logging validation batch...')
    valid_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('build Valid batch...')
    #idx = np.random.choice(list(range(len(test_dataset))), size=params.test_size, replace=False)
    
    valid_batch_text = []
    valid_batch_mel = []
    filepaths = []
    for filepath_and_text in valid_dataset.filepaths_and_text:
        # Each entry of filepaths_and_text is a [filepath, text_content] list.
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        mel = valid_dataset.get_mel(filepath)

        filepaths.append(filepath)
        valid_batch_text.append(text)  # [{'y': mel, 'x': text}, {'y': mel, 'x': text}, {'y': mel, 'x': text}]
        valid_batch_mel.append(mel)

    print('output original mel spectrogram and text')
    print('all mel spectrogram is written in a file')
    print('all text is written in a file')
    if not os.path.exists(gt_dir):
        os.makedirs(f'{gt_dir}')
    if not os.path.exists(cvt_dir):
        os.makedirs(f'{cvt_dir}')

    with open(args.file, 'w') as text_file:
        for i, item in enumerate(valid_batch_text):
            text_file.write(f"{valid_batch_text[i]}\n")
    
    texts = valid_batch_text       

    print('move the.wav file from LJSpeech dataset to eval/original directory')
    for i, filepath in enumerate(filepaths):
        shutil.copy(filepath, f'eval/original/output_{i}.wav')

    save_mel_spectrograms_to_file(valid_batch_mel, f'{gt_dir}')

    print('get checkpts paths')
    checkpoint_files = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir) if file.endswith('.pt')]
    checkpoint_files = sorted(checkpoint_files, key=get_integer_part)

    print('Initializing HiFi-GAN as vocoder')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    print('Initialize GradTTS MODEL')
    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)


    for i in range(0,len(checkpoint_files),epoch_interval):
        y_mel = []
        #get integer parts of the file
        checkpoint_name = _get_basename(checkpoint_files[i])
        number_part = get_integer_part(checkpoint_name)
        index = int(number_part)

        generator.load_state_dict(torch.load(f'{checkpoint_files[i]}', map_location=lambda loc, storage: loc))
        _ = generator.cuda().eval()
        print(f'Doing the {index}st epoch')
        os.makedirs(f'{cvt_dir}/Epoch_{index}')
        with torch.no_grad():
           for i, text in enumerate(texts):
                    # convert word to phonemes:
                    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                           stoc=False, length_scale=0.91)
                    y_mel.append(y_dec) # [put all mel tensor from text file into a list]

                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    write(f'./eval/converted/Epoch_{index}/output_{i}.wav', 22050, audio)
                    pt_to_pdf(y_dec.cpu().squeeze(0), f'./eval/converted/Epoch_{index}/dec_{i}.pdf')
                    pt_to_pdf(y_enc.cpu().squeeze(0), f'./eval/converted/Epoch_{index}/enc_{i}.pdf')
        save_mel_spectrograms_to_file(y_mel, f'{cvt_dir}/Epoch_{index}')

