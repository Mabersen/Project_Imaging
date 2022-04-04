"""
Code to select and save a random subsample of the dataset.

To use: Change the paths and the subsample factor and run the file. The subsampled_path will 
be created during execution and the selected subsampled images saved in it.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import os
import random
import shutil
# change these as necessary
original_path = r"C:\Users\maber\Desktop\ari\8P361\train+val_sub_75\train"
subsampled_path = r"C:\Users\maber\Desktop\ari\8P361\train+val_sub_25\train"
# change this to whatever you want
subsample_factor = 1/3
try:
    os.makedirs(subsampled_path)
    os.mkdir(subsampled_path + r"\0")
    os.mkdir(subsampled_path + r"\1")
except OSError:
    print("Directory already exists, please choose a different directory for your subsampled data.")
    raise OSError
files_0 = os.listdir(original_path + r'\0')
files_1 = os.listdir(original_path + r'\1')
sample_0 = random.sample(files_0, int(subsample_factor*len(files_0)))
sample_1 = random.sample(files_1, int(subsample_factor*len(files_1)))
for file in sample_0:
    shutil.copy2(original_path + r'\0\\' + file, subsampled_path + r'\0')
for file in sample_1:
    shutil.copy2(original_path + r'\1\\' + file, subsampled_path + r'\1')





