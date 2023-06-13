import os
import sys
from glob import glob

assert len(sys.argv)==2, "Pass scene name"

files = os.listdir('/export/work/sanskar/nerfactor_without_neus_data/input/' + sys.argv[1])
files = [x for x in files if x.startswith('train_')]
a = len(files)

print(a-1)