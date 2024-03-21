import os
import torch
from torch.utils.data import DataLoader
import numpy as np

import params

from data import TextMelDataset, TextMelBatchCollate

valid_filelist_path = params.valid_filelist_path
random_seed = params.seed
cmudict_path = params.cmudict_path
add_blank = params.add_blank


if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                            n_fft, n_feats, sample_rate, hop_length,
                                            win_length, f_min, f_max)


    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                                           n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                                           n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    
    

