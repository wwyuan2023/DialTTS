#coding: utf-8

import os, sys

import numpy as np
import matplotlib.pyplot as plt


dim = int(sys.argv[1])
out_file = sys.argv[2]

in_files = sys.argv[3:]
n = len(in_files)

for i in range(1, n+1):
    y = np.fromfile(in_files[i-1], dtype=np.float32).reshape(-1, dim).T # (d,T)
    print(">>>>>", i, y.shape)
    plt.subplot(n, 1, i)
    plt.pcolor(y)
plt.tight_layout()
plt.savefig(out_file)
plt.close()

