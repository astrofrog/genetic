import os
import glob

import atpy
import numpy as np
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mpl

from genetic import Genetic


# Here we define the parsing function that tells the genetic code how to read
# in the values from a parameter file. In this case, we assume the parameter
# file defines values like:
#
# mdisk = 1.
# mdot = 1.e-2
# ltot = 100.

def parser(line):
    key, value = line.split('=')
    return key.strip(), value.strip()


# The following is the class used to run the models. It requires a single
# method, 'run', that should take (in addition to self), a parameter file
# name, a directory where to write the model to, and the model name. It is
# also possible to re-define __init__ so as to be able to pass arguments that
# remain valid for the duration of the run (see PolyFitter for an example).

class PolyModel(object):

    def __init__(self):
        pass

    def run(self, par_file, model_dir, model_name):

        # First we read in the parameter file
        f = file(par_file, 'r')
        a = float(f.readline().split('=')[1].strip())
        b = float(f.readline().split('=')[1].strip())
        c = float(f.readline().split('=')[1].strip())
        d = float(f.readline().split('=')[1].strip())

        # Then we compute the model
        x = np.linspace(-2., 2., 100)
        y = a * x ** 3 + b * x ** 2 + c * x + d

        # Finally we write it out to a file
        np.savetxt(model_dir + str(model_name) + '.poly', zip(x, y), fmt="%.3f")


# The following is the class used to fit the models. It requires a single
# method, 'run', that takes the directory containing the models, the file to
# output the fitting results to, and the directory to output the plots to. In
# addition, this example demonstrates how __init__ can be used to define the
# data once and for all

class PolyFitter(object):

    def __init__(self, datafile):
        data = np.loadtxt(datafile, dtype=[('x',float),('y',float)])
        self._x = data['x']
        self._y = data['y']

    def run(self, models_dir, output_file, plots_dir):

        model_name = []
        chi2 = []

        # Loop over all models we can find in the models directory. As defined
        #in PolyModel, we know all the model files have a .poly extension.
        for model_file in glob.glob(os.path.join(models_dir,'*.poly')):

            # Read in model
            model_tmp = np.loadtxt(model_file, dtype=[('x',float),('y',float)])

            # Create interpolating function
            model = interp1d(model_tmp['x'], model_tmp['y'])

            # Calculate the chi^2 value - model(self._x) interpolates the
            # model to the data x values on the fly
            chi2.append(np.sum((model(self._x)-self._y)**2))

            # Figure out the model name from the model filename
            name = model_file.split('/')[-1].replace('.poly','')

            # Add the mode name to a list (which will be output to the table)
            model_name.append(name)

            # Make a plot of the fit
            fig = mpl.figure()
            ax = fig.add_subplot(1,1,1)
            ax.scatter(self._x, self._y)
            ax.plot(model_tmp['x'], model_tmp['y'])
            fig.savefig(plots_dir + name + '.png')

        # Write output chi^2 table
        t = atpy.Table()
        t.add_column('model_name', model_name, dtype='|S30')
        t.add_column('chi2', chi2)
        t.write(output_file)

# Above, we've defined the functions and classes. Now we can finally use them.
# First, we create an instance of the class to run the models...

poly_model = PolyModel()

# ... then we create an instance of the class to fit the data, and give it the
# name of the datafile, which is required by __init__.

model_fitter = PolyFitter('data_example1')

# Finally, we can set up the genetic algorithm itself. The first argument is
# the number of models to run in the first generation. The econd argument is
# the output directory. The third argument is a template for the parameter
# file required by PolyModel, which shoud resemble a normal file, but should
# contain VAR for every parameter that is varied. Finally, the fourth
# argument should be the file describing the ranges of parameters. The
# remaining parameters are optional: fraction_output is the fraction of
# models to compute in each generation after the first, and fraction_mutation
# gives the fraction of mutations, the remaining being crossovers.

g = Genetic(100, 'models_example1', 'template.par', 'example1.conf',
            fraction_output=0.1, fraction_mutation=0.5)

# ... and we can loop over generations

for generation in range(1,50):

    # The first step is to create a directory of the form g????? where ?????
    # is the generation number, e.g. g00001 for the first generation. Inside
    # this directory, initialize() will create directories for plots and
    # models

    g.initialize(generation)

    # We then create a parameter table. This is a single FITS table that
    # contains all the randomly sampled values for all parameters for all the
    # models to run.

    g.make_par_table(generation)

    # The following then reads the parameter table, and for each, uses the
    # parameter file template specified when initializing Genetic() to create
    # one parameter file per model.

    g.make_par_indiv(generation, parser)

    # Finally, we can compute the models, passing it the necessary class
    # instance...

    g.compute_models(generation, poly_model)

    # ... and we compute the chi^2 fits and output a FITS table with all the
    # model names and chi^2 values

    g.compute_fits(generation, model_fitter)
