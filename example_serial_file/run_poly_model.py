#!/usr/bin/env python

import sys
import string

import numpy as np

if len(sys.argv[1:]) != 3:
    print "Usage: ./run_poly_model.py par_file model_dir model_name"
    print "Arguments given: " + string.join(sys.argv[1:], " ")
    sys.exit(22)

par_file, model_dir, model_name = sys.argv[1:]

f = file(par_file, 'r')
a = float(f.readline().split('=')[1].strip())
b = float(f.readline().split('=')[1].strip())
c = float(f.readline().split('=')[1].strip())
d = float(f.readline().split('=')[1].strip())

x = np.linspace(-2.,2.,100)
y = a*x**3 + b*x**2 + c*x + d

np.savetxt(model_dir + str(model_name) + '.poly', zip(x, y), fmt="%.3f")
