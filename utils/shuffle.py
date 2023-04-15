import os
import random
import shutil
list=['potato_healthy','tomato_healthy']
dir = "/Users/bhanuprakash/Documents/AI/project/idata/Image Dataset/ImageDataset/valid/"
dest_dir = "/dataset_2_class/train/healthy/"
for i in list:
    src_dir=dir+i
    jpg_files = [f for f in os.listdir(src_dir) if f.endswith('.JPG')]
    random.shuffle(jpg_files)
    print(jpg_files)
    for jpg_file in jpg_files:
        src_file = os.path.join(src_dir, jpg_file)
        dest_file = os.path.join(dest_dir, jpg_file)
        shutil.copy(src_file, dest_file)
