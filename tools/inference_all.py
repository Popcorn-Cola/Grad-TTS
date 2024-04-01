import argparse
import json
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys, os, re
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import matplotlib.pyplot as plot

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def pt_to_pdf(pt,pdf,vmin=-12.5,vmax=0.0):
  spec=pt
  fig=plot.figure(figsize=(20,4),tight_layout=True)
  subfig=fig.add_subplot()
  image=subfig.imshow(spec,cmap="jet",origin="lower",aspect="equal",interpolation="none",vmax=vmax,vmin=vmin)
  fig.colorbar(mappable=image,orientation='vertical',ax=subfig,shrink=0.5)
  plot.savefig(pdf,format="pdf")
  plot.close()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint_dir', type=str, required=True, help='path to checkpoint directory of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    args = parser.parse_args()

    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None


    #import text file and cmudict
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    print('get file index')
    checkpoint_list, ckpts_id = find_files(args.checkpoint_dir)

    print('Initializing HiFi-GAN as vocoder')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    print('Initialize GradTTS MODEL')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)


    for index, checkpoint in enumerate(checkpoint_list):
        generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
        _ = generator.cuda().eval()

        Epoch_ind = index
        with torch.no_grad():
            for i, text in enumerate(texts):
                print(f'Synthesizing {i} text...', end=' ')
                #convert word to phonemes:
                x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                write(f'./out/Epoch_{Epoch_ind}.wav', 22050, audio)

                pt_to_pdf(y_enc.cpu().squeeze(0), f'./out/Epoch_{Epoch_ind}_Encoder.pdf')
                pt_to_pdf(y_dec.cpu().squeeze(0), f'./out/Epoch_{Epoch_ind}_Decoder.pdf')

    print('Done. Check out `out` folder for samples.')
