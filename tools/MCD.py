import os
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
from fastdtw import fastdtw
from scipy import spatial

def read_tensor(mel_folder, num_tensor): #get input 1.folder containing files of mel .pt  2. number of mel .pt files in that folder 
    tensor_list = []
    for i in range(num_tensor):
        tensor_list.append(torch.load(f'{mel_folder}/{i}.pt'))
    return tensor_list                   #return a list containing tensors

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--cvt_directory', type=str, required=False, default='/exp/exp4/acp23xt/TAN-Grad-TTS/eval/converted', help='folder containing converted data')
    parser.add_argument('-a', '--gt_directory', type=str, required=False,
                        default='/exp/exp4/acp23xt/TAN-Grad-TTS/eval/original',
                        help='folder containing original data')
    parser.add_argument('-o', '--out_directory', type=str, required=False,
                        default='/exp/exp4/acp23xt/TAN-Grad-TTS/eval/MCD',
                        help='folder containing converted data')
    parser.add_argument('-n', '--num_letter', type=int, required=False, default=6,
                        help='number of prefix letter of folder name. The folder contains result of each epoch')
    args = parser.parse_args()

    out_directory = args.out_directory
    cvt_directory = args.cvt_directory
    gt_directory = args.gt_directory
    num_letter = args.num_letter
    os.makedirs(out_directory, exist_ok=True)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('please ensure that mel tensor files are stored in i.pt format')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('read the ground truth mel')
    tensor_num = int(0)
    for file in os.listdir(gt_directory):
        if file.endswith('.pt'):
            tensor_num = tensor_num+1

    gt_tensor= read_tensor(f"{gt_directory}", tensor_num)

    direcs = os.listdir(cvt_directory)# 'direcs' is a list containing name of sub-folders in a folder
    direc_abspaths = [os.path.join(cvt_directory, direc) for direc in direcs]

    MCD_mean = {}
    MCD_std = {}
    for index, direc_abspath in enumerate(direc_abspaths):
        cvt_tensor = read_tensor(f"{direc_abspath}", tensor_num)
        mcd_l = np.empty([])
        for i in range(tensor_num):
            # DTW

            # print(f'cvt_tensor[i].cpu() is {cvt_tensor[i].cpu().numpy().squeeze()}')
            # print(np.shape(cvt_tensor[i].cpu().numpy().squeeze()))
            # print(np.shape(gt_tensor[i].cpu().numpy().squeeze()))
            cvt_np = cvt_tensor[i].cpu().numpy().squeeze().T
            gt_np  = gt_tensor[i].cpu().numpy().squeeze().T
            _, path = fastdtw(cvt_np, gt_np, dist=spatial.distance.euclidean)
            # fs, signal = wavfile.read(f"{direc}/{i}.wav")
            twf = np.array(path).T
            cvt_np_dtw = cvt_np[twf[0]]
            gt_np_dtw = gt_np[twf[1]]
            diff2sum = np.sum((cvt_np_dtw - gt_np_dtw) ** 2, 1)
            mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
            mcd_l = np.append(mcd_l, mcd)
        MCD_mean[direcs[index]] = np.mean(mcd_l)
        MCD_std[direcs[index]] = np.std(mcd_l)

    sorted_MCD_keys = sorted(MCD_mean.keys(), key=lambda x: int(x[num_letter:]))
    
    MCD_mean_list = [MCD_mean[key] for key in sorted_MCD_keys]
    MCD_std_list = [MCD_std[key] for key in sorted_MCD_keys]
    # with open(f"{cvt_directory}/MCD_f0.txt", 'w') as file:
    #     for i in range(MCD):
    #         file.write(str(MCD[i]) + ' ')
    #         file.write('\n')
    #         file.write(str(F0[i]) + ' ')
    with open(f"{out_directory}/MCD_mean.txt", "w") as f:
        for key in sorted_MCD_keys:
            f.write(f'{key}  MCD is {MCD_mean[key]}/n')
            
    with open(f"{out_directory}/MCD_std.txt", "w") as f:
        for key in sorted_MCD_keys:
            f.write(f'{key}  MCD is {MCD_std[key]}/n')

    my_list = list(range(0, 3100, 100))

    plt.plot(my_list, MCD_mean_list)
    plt.xlabel('Epochs')
    plt.ylabel('MCD')
    plt.title('MCD')
    plt.savefig(f"{out_directory}/MCD.png")

    plt.plot(my_list, MCD_std_list)
    plt.xlabel('Epochs')
    plt.ylabel('MCD')
    plt.title('MCD')
    plt.savefig(f"{out_directory}/MCD.png")