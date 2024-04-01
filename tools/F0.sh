export CUDA_VISIBLE_DEVICES=3

Converted_directory="/exp/exp4/acp23xt/TAN-Grad-TTS/eval/converted"
Output_directory="/exp/exp4/acp23xt/TAN-Grad-TTS/ex"


sorted_folder=$(python sorted.py -d $Converted_directory)

IFS=' '

folder_list=($sorted_folder)

echo "directory to be converted is, $folder_list, printed"  

for folder in "${folder_list[@]}"; do
	echo "Now is processing $folder"	
	python F0.py /exp/exp4/acp23xt/TAN-Grad-TTS/eval/original "$folder" --outdir "$Output_directory"
	done

python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/mean_log_f0_rmse.txt -s 0 -i 100 -t F0 -x Epoch -y F0_rmse_mean -o $Output_directory/F0_rmse_mean.png 


python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/std_log_f0_rmse.txt -s 0 -i 100 -t F0 -x Epoch -y F0_rmse_std -o $Output_directory/F0_rmse_std.png
