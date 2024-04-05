export CUDA_VISIBLE_DEVICES=3

python eval_all.py -c 'logs/2024-03-19' -i 100 -g 'test/ground_truth' -z 'test/converted' -m 'WAVPDFMEL' -o 'resources/filelists/ljspeech/test.txt'
