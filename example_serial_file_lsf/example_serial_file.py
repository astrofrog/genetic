import numpy as np
from scipy.interpolate import interp1d
import atpy
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mpl

from genetic import Genetic

def parser(line):
    return line.split('=')[0].strip(), line.split('=')[1].strip()


class PolyFitter(object):

    def __init__(self, datafile):
        data = np.loadtxt(datafile, dtype=[('x',float),('y',float)])
        self._x = data['x']
        self._y = data['y']

    def run(self, models_dir, output_file, plots_dir):

        model_name = []
        chi2 = []

        for model_file in glob.glob(os.path.join(models_dir,'*.poly')):
            model_tmp = np.loadtxt(model_file, dtype=[('x',float),('y',float)])
            model = interp1d(model_tmp['x'], model_tmp['y'])
            chi2.append(np.sum((model(self._x)-self._y)**2))
            name = model_file.split('/')[-1].replace('.poly','')
            model_name.append(name)

            fig = mpl.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self._x, self._y)
            ax.plot(model_tmp['x'], model_tmp['y'])
            fig.savefig(plots_dir + name + '.png')

        t = atpy.Table()
        t.add_column('model_name', model_name, dtype='|S30')
        t.add_column('chi2', chi2)
        t.write(output_file)

poly_model = os.path.abspath('submit_job_lsf.py')
model_fitter = PolyFitter('data_example_serial_file')

g = Genetic(100, 'models_example_serial_file', 'template.par', 'example_serial_file.conf', existing=False, mode='serial_file')

for generation in range(1,50):

    g.initialize(generation)
    g.make_par_table(generation)
    g.make_par_indiv(generation, parser)
    g.compute_models(generation, poly_model)
    g.compute_fits(generation, model_fitter)
