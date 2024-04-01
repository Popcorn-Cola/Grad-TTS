import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir', type=str, required=True,
                        help='source directory that contains all results')
    parser.add_argument('-d', '--dest_dir', type=str, required=True,
                        help='destination directory of output')
    parser.add_argument('-t', '--dest_list', nargs='+', required=True,
                        help='destination filename')
    
    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    dest_list = args.dest_list
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dirs = os.listdir(source_dir)

    for dir in dirs:
        for dest in dest_list:
            source_file_path = os.path.join(source_dir, dir, dest)
            #parent_directory_name = os.path.basename(source_file_path)
            destination_file_path = os.path.join(dest_dir, f'{dir}_{dest}')
            shutil.copy(source_file_path, destination_file_path)
