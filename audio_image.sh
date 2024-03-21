export CUDA_VISIBLE_DEVICES="1" 

save_directory="$go_ex/TAN-Grad-TTS/result/Audio_and_image"
checkpts_directory="$go_ex/TAN-Grad-TTS/logs/2024-03-19"

files=$(ls "$checkpts_directory" | grep -E 'grad_[0-9]+\.pt' | sed -E 's/grad_([0-9]+)\.pt/\1/g' | sort -n)

for i in $files; do
	python inference.py -f 'resources/filelists/synthesis.txt' -c "$checkpts_directory/grad_$i.pt"
done	


