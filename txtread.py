# Load heroines coordinates
import torch
import numpy as np

f = open("mudo_center.txt", 'r')
lines = f.readlines()
frame = 0

for i in range(20):
    line = lines[frame].strip().split(', ')
    frame += 1
    # print(line)
    if line[1] == 'None':
        continue
    temp = torch.from_numpy(np.array(line[1:]).astype(np.float32))



