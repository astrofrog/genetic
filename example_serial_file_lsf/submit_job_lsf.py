#!/usr/bin/env python

import os
import sys
import string
import subprocess

import numpy as np

if len(sys.argv[1:]) != 3:
    print "Usage: ./run_poly_model.py par_file model_dir model_name"
    print "Arguments given: " + string.join(sys.argv[1:], " ")
    sys.exit(22)

par_file, model_dir, model_name = sys.argv[1:]

TEMPLATE = \
"""#!/bin/bash
#BSUB -u trobitaille@cfa.harvard.edu
#BSUB -q normal_serial
#BSUB -J $JOBID
#BSUB -o $LOGFILE
#BSUB -e $ERRFILE
python run_poly_model.py $ARGS
"""

log_file = os.path.join(model_dir, model_name + '.lsf.log')
err_file = os.path.join(model_dir, model_name + '.lsf.err')

if os.path.exists(log_file):
    os.remove(log_file)
if os.path.exists(err_file):
    os.remove(err_file)

submission_script = TEMPLATE.replace('$JOBID', model_name)
submission_script = submission_script.replace('$LOGFILE', log_file)
submission_script = submission_script.replace('$ERRFILE', err_file)
submission_script = submission_script.replace('$ARGS', par_file + ' ' + model_dir + ' ' + model_name)

import tempfile
f = tempfile.NamedTemporaryFile()
f.write(submission_script)
f.flush()
p = subprocess.Popen('bsub < %s' % f.name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

# Wait for submission to complete
p.wait()

# Close submission file
f.close()

# Print out stdout and stderr
stdout = p.stdout.read()
stderr = p.stderr.read()

# Check if any errors occurred
if stderr.strip() != "":
    print "An error occurred when submitting the job"
    sys.exit(1)

# Extract job ID from submission stdout
p1 = stdout.index('<')
p2 = stdout.index('>', p1)
id = stdout[p1+1:p2]
print "Submission ID: ", id

# The run is finished when the log file appears
import time
while True:
    if os.path.exists(log_file):
        break
    time.sleep(10)
