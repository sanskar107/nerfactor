import os
import sys
from glob import glob

assert len(sys.argv)==2, "Pass scene name"

files = os.listdir('/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/input/' + sys.argv[1])
files = [x for x in files if x.startswith('train_')]
a = len(files)

path = '/export/share/projects/svbrdf/data/generated/colmap/' + sys.argv[1] + '/out_im*.png'
files = glob(path)
b = len(files)
assert a==b, "invalid input"

print(a-1)
