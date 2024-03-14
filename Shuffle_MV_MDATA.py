import csv
import random
import os

def reorder_lines(input_file, output_file, index_list):
    with open(input_file, 'r', newline='') as f_in:
        reader = csv.reader(f_in, delimiter=' ', quotechar='|')
        for row in reader:
            rows.append(row)

    sorted_lines = [rows[i] for i in index_list]
    
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(sorted_lines)

def file_generate(input_file, output_file, index_a, index_b):
    with open(input_file, 'r', newline='') as f_in:
        reader = csv.reader(f_in)
        lines = list(reader)

    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(lines[index_a: index_b])

# Example usage

rows = []
input_file = '/store/store4/data/LJSpeech-1.1/metadata.csv'
output_folder = '/store/store4/data/LJSpeech-TAN'

output_file = os.path.join(output_folder, 'ALL_METADATA.csv')
train_file =  os.path.join(output_folder, 'train.csv')
dev_file   =  os.path.join(output_folder, 'dev.csv')
test_file   =  os.path.join(output_folder, 'test.csv')

my_list = list(range(13100)) #13100
random.seed(40)
random.shuffle(my_list)
reorder_lines(input_file, output_file, my_list)
file_generate(output_file , train_file,  0, 11947)
file_generate(output_file , dev_file, 11947, 12435)
file_generate(output_file , test_file, 12435, 12530)

print("CSV file lines reordered and saved to:" , output_folder )

print("All files moved successfully")
