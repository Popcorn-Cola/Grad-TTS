export CUDA_VISIBLE_DEVICES=1

Converted_directory="/exp/exp4/acp23xt/TAN-Grad-TTS/eval/converted"

filenames=($(find "$Converted_directory" -mindepth 1 -maxdepth 1 -type d))

sorted_filenames=($(printf "%s\n" "${filenames[@]}" | grep -oE '[0-9]+' | sort -n | while read -r num; do grep "file$num.txt" <<< "${filenames[@]}"; done))

for element in "${sorted_filename[@]}"; do
    echo "$element"
done
