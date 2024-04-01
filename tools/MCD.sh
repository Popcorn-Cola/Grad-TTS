export CUDA_VISIBLE_DEVICES=1

Converted_directory="/exp/exp4/acp23xt/TAN-Grad-TTS/eval/converted"
Output_directory="/exp/exp4/acp23xt/TAN-Grad-TTS/ex"

sorted_folder=$(python sorted.py -d $Converted_directory)

IFS=' '

folder_list=($sorted_folder)

for folder in "${folder_list[@]}"; do
        echo "$folder"
        python MCD_2.py /exp/exp4/acp23xt/TAN-Grad-TTS/eval/original "$folder" --outdir "$Output_directory"
        done

        python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/mean_mcd.txt -s 0 -i 100 -t MCD -x Epoch -y mean_mcd -o $Output_directory/mean_mcd.png


	python line_chart.py -p /exp/exp4/acp23xt/TAN-Grad-TTS/ex/std_mcd.txt -s 0 -i 100 -t MCD -x Epoch -y std_mcd -o $Output_directory/std_mcd.png

