# Build-in Python modules
import os
import glob
import string
import random

# Third-party modules
import atpy
import numpy as np

# Sub-modules from this package
from .utils import create_dir
from .select import select

# Defines how many samples to try before giving up
N_MAX_SAMPLE = 10000


class Genetic(object):

    def __init__(self, n_models, output_dir, template, configuration,
                 existing=False, fraction_output=0.1, fraction_mutation=0.5):
        """
        The Genetic class is used to control the SED fitter genetic algorithm

        Parameters
        ----------

        n_models: int
            Number of models to run in the first generation, and to keep in
            subsequent generations

        output_dir: str
            The directory in which to output all the models

        template: str
            The template parameter file. This is a file that looks like the
            file that needs to be passed to the running function the user
            specifies, but with any variable value set to VAR.

        configuration: str
            The configuration file that describes how the parameters should be
            sampled. This file should contain four columns:
                * The name of the parameter (no spaces)
                * Whether to sample linearly ('linear') or logarithmically ('log')
                * The minimum value of the range
                * The maximum value of the range

        existing: bool, optional
           Whether to keep any existing model directory (useful when starting
           from a later generation). If this option is set to false,
           `output_dir` will be deleted.

        fraction_output: float, optional
           Fraction of models to add to and remove from the pool at each
           generation

        fraction_mutation: float, optional
           Fraction of children that are mutations (the remainder being crossovers)
        """

        # Read in parameters
        self.n_models = n_models
        self._models_dir = output_dir

        # Read in template parameter file
        self._template = open(template, 'rb').readlines()

        # Read in configuration file
        self.parameters = {}
        for line in file(configuration, 'rb'):
            if not line.strip() == "":
                name, sampling_mode, vmin, vmax = string.split(line)
                self.parameters[name] = {'mode': sampling_mode,
                                         'min': float(vmin),
                                         'max': float(vmax)}

        # Set genetic parameters
        self._fraction_output = fraction_output
        self._fraction_mutation = fraction_mutation

        # Create output directory
        if not existing:
            create_dir(self._models_dir)

    def initialize(self, generation):
        """
        Initialize the directory structure for the generation specified.

        Parameters
        ----------
        generation : int
            The generation to initialize the directories for.
        """
        create_dir(self._generation_dir(generation))
        create_dir(self._model_dir(generation))
        create_dir(self._parameter_dir(generation))
        create_dir(self._plots_dir(generation))

    def make_par_table(self, generation, validate=lambda x: True):
        """
        Creates a table of models to compute for the generation specified.

        If this is the first generation, then the parameters are sampled in an
        unbiased way within the ranges specified by the user. Otherwise, this
        method uses results from previous generations to determine which
        models to run.

        Parameters
        ----------
        generation : int
            The generation to make a parameter table for.
        validate : function, optional
            Optionally, it is possible to specify a function that determines
            whether a given set of parameters is possible. By default, all
            models sampled are possible, but in some cases, it may be
            desirable to disallow certain combinations of parameters (e.g.
            unphysical ones).
        """

        print "[genetic] Generation %i: making parameter table" % generation

        t = atpy.Table()

        if generation == 1:

            # If this is the first generation, then we just sample within
            # ranges, and don't do any mutations/crossovers.

            print "[genetic] Generation %i: initializing parameter file" % generation

            # Create model names column
            t.add_column('model_name', ["g1_" + str(i) for i in range(self.n_models)], dtype='|S30')

            # Create empty columns in table for each parameter
            for par_name in self.parameters:
                t.add_empty_column(par_name, dtype=float)

            # Loop over each model and sample the parameters. The loop
            # variable i is used to access the relevant row in the table.
            for i in range(self.n_models):

                # This loop is only useful when specifying a validation
                # function. The purpose is that if the model is invalid
                # according to the validation function, we need to sample a
                # new set of parameters.
                for sample in range(N_MAX_SAMPLE):

                    # Loop over parameters
                    for par_name in self.parameters:

                        # Get settings from configuration file
                        mode = self.parameters[par_name]['mode']
                        vmin = self.parameters[par_name]['min']
                        vmax = self.parameters[par_name]['max']

                        # Sample value
                        if mode == "linear":
                            value = random.uniform(vmin, vmax)
                        elif mode == "log":
                            value = 10. ** random.uniform(np.log10(vmin), np.log10(vmax))
                        else:
                            raise Exception("Unknown mode: %s" % mode)

                        # Add value to table
                        t.data[par_name][i] = value

                    # If the model validates (which it always does if a
                    # validation function was not provided), we can continue.
                    if validate(t.data[i]):
                        break

                if sample == N_MAX_SAMPLE - 1:
                    raise Exception("Could not sample a valid model after {:d} tries".format(N_MAX_SAMPLE))

        else:

            # If we are not on the first generation, things are a little more
            # evolved because we need to read in the results from previous
            # generations and use those to inform the current sampling.

            # Decide how many models to sample in this generation
            n_output = int(self.n_models * self._fraction_output)

            # Create model names column
            t.add_column('model_name', ["g%i_%i" % (generation, i) for i in range(n_output)], dtype='|S30')

            # Create empty columns in table for each parameter
            for par_name in self.parameters:
                t.add_empty_column(par_name, dtype=float)

            # Read in parameter tables from the first generation
            par_table = atpy.Table(self._parameter_table(1),
                                   verbose=False)

            # Read in the tables from subsequent generations, and append
            for g in range(2, generation):
                par_table.append(atpy.Table(self._parameter_table(g),
                                 verbose=False))

            # Read in fitter results from first generation
            chi2_table = atpy.Table(self._fitting_results_file(1),
                                    verbose=False)

            # Read in the tables from subsequent generations, and append
            for g in range(2, generation):
                chi2_table.append(atpy.Table(self._fitting_results_file(g),
                                  verbose=False))

            # Sort from best to worst-fit chi^2
            chi2_table.sort('chi2')

            # Truncate the table to the n_models first models
            chi2_table = chi2_table.rows(range(self.n_models))

            print "[genetic] Generation %i: best fit so far: %s with chi^2=%g" % (generation, chi2_table['model_name'][0].strip(), chi2_table['chi2'][0])

            # Initialize list to keep track of which models were used in the
            # selection
            selected = []

            # Initialize counters for mutations and crossovers
            mutations = 0
            crossovers = 0

            # Open log file to keep track of mutations and crossovers
            logfile = open(self._log_file(generation), 'wb')

            # Loop over the models to compute
            for i in range(n_output):

                # Select whether to do crossover or mutation

                if(random.random() > self._fraction_mutation):  # crossover

                    crossovers += 1

                    # This is the tournament selection. We use the list of
                    # chi^2 values and the tournament selection to pick two
                    # models to combine. For more details, see:
                    # http://en.wikipedia.org/wiki/Tournament_selection
                    im1, im2 = select(chi2_table.chi2, n=2, k_frac=0.1, p=0.9)

                    # The above returned the index of the model, so we now
                    # extract the names
                    m1 = chi2_table.model_name[im1]
                    m2 = chi2_table.model_name[im2]

                    # Add an entry to the log file
                    logfile.write('g%s_%i = crossover of %s and %s\n' % (generation, i, m1, m2))

                    # Keep track of which models have been used
                    selected.append(im1)
                    selected.append(im2)

                    # The following finds the row in the parameter table where
                    # the model name is equal to the selected model, for both
                    # models.
                    par_m1 = par_table.row(np.char.strip(par_table.model_name) == m1.strip())[0]
                    par_m2 = par_table.row(np.char.strip(par_table.model_name) == m2.strip())[0]

                    # As for the first generation, this loop is only useful if
                    # the validation function is customied. The purpose is to
                    # loop until a valid model is found.
                    for sample in range(N_MAX_SAMPLE):

                        # Loop over parameters
                        for par_name in self.parameters:

                            # Get the parameter values for both models
                            par1 = par_m1[par_name]
                            par2 = par_m2[par_name]
                            mode = self.parameters[par_name]['mode']

                            # Sample a random fraction with which to combine the models
                            xi = random.random()

                            # Sample value
                            if mode == "linear":
                                value = par1 * xi + par2 * (1. - xi)
                            elif mode == "log":
                                value = 10. ** (np.log10(par1) * xi + np.log10(par2) * (1. - xi))
                            else:
                                raise Exception("Unknown mode: %s" % mode)

                            # Add value to table
                            t.data[par_name][i] = value

                        # If the model validates (which it always does if a
                        # validation function was not provided), we can continue.
                        if validate(t.data[i]):
                            break

                    if sample == N_MAX_SAMPLE - 1:
                        raise Exception("Could not sample a valid model after {:d} tries".format(N_MAX_SAMPLE))

                else:  # mutation

                    mutations += 1

                    # Use tournament selection to select a single model
                    im1 = select(chi2_table.chi2, n=1, k_frac=0.1, p=0.9)[0]

                    # Find the model name
                    m1 = chi2_table.model_name[im1]

                    # Add an entry to the log file
                    logfile.write('g%s_%i = mutation of %s\n' % (generation, i, m1))

                    # Keep track of which models have been used
                    selected.append(im1)

                    # Pick a parameter at random to mutate (this returns the name of the parameter)
                    mutation = random.choice(self.parameters.keys())

                    # Extract the row from the parameter table
                    par_m1 = par_table.row(np.char.strip(par_table.model_name) == m1.strip())[0]

                    # As for the first generation, this loop is only useful if
                    # the validation function is customied. The purpose is to
                    # loop until a valid model is found.
                    for sample in range(N_MAX_SAMPLE):

                        # Loop over parameters
                        for par_name in self.parameters:

                            # Get the parameter value for this model
                            value = par_m1[par_name]

                            # If the parameter is the one to mutate, then
                            # sample a new value as in the first generation
                            if par_name == mutation:

                                mode = self.parameters[par_name]['mode']
                                vmin = self.parameters[par_name]['min']
                                vmax = self.parameters[par_name]['max']

                                # Sample value
                                if mode == "linear":
                                    value = random.uniform(vmin, vmax)
                                elif mode == "log":
                                    value = 10. ** random.uniform(np.log10(vmin), np.log10(vmax))
                                else:
                                    raise Exception("Unknown mode: %s" % mode)

                            # Add value to table
                            t.data[par_name][i] = value

                        # If the model validates, then break the loop
                        if validate(t.data[i]):
                            break

                    if sample == N_MAX_SAMPLE - 1:
                        raise Exception("Could not sample a valid model after {:d} tries".format(N_MAX_SAMPLE))

            # Close the log file
            logfile.close()

            # Print some statistics
            print "          Mutations  : " + str(mutations)
            print "          Crossovers : " + str(crossovers)

            # Make a plot showing the number of times each model was
            # selected. We use the OO interface to matplotlib to avoid memory
            # leaks.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(selected, 50)
            canvas.print_figure(self._sampling_plot_file(generation))

        # Write out the parameter table
        t.write(self._parameter_table(generation), verbose=False)

    def make_par_indiv(self, generation, parser, interpreter=None):
        """
        For the generation specified, will read in the parameters.fits file
        and the parameter file template, and will output individual parameter
        files to the par/ directory for that given generation.

        Parameters
        ----------
        generation : int
            The generation to make a parameter table for.
        parser : function
            A function that given a line from the parameter file will return
            a (key, value) tuple
        interpreter : function, optional
            A function that given a parameter name and a dictionary of
            parameter values, will determine the actual value to use (useful
            for example if several parameters are correlated).
        """

        print "[genetic] Generation %i: making individual parameter files" % generation

        # Read in parameter table and construct dictionary
        table = atpy.Table(self._parameter_table(generation), verbose=False)
        par_table = [dict(zip(table.names, table.row(i)))
                        for i in range(len(table))]

        # Find static parameters
        static = {}
        for line in self._template:
            name, value = parser(line)
            if value != 'VAR' and value is not None:
                static[name] = value

        # Cycle through models and create a parameter file for each
        for model in par_table:

            # Create new parameter file
            model_name = model['model_name'].strip()
            f = open(self._parameter_file(generation, model_name), 'wb')

            # Cycle through template lines
            for line in self._template:

                # Get parser to return the (key, value) pair
                name, value = parser(line)

                if value == 'VAR':

                    # If the parameter is variable, then we find the value
                    # from the parameter file

                    # If an interpreter was specified, we have to use that
                    if interpreter:
                        value = interpreter(generation, name, dict(model.items() + static.items()))
                    else:
                        value = model[name]

                    f.write(line.replace('VAR', str(value)))

                else:

                    # If the parameter is not variable, we just output the
                    # line unmodified

                    f.write(line)

            f.close()

    def compute_models(self, generation, model):
        """
        Compute all the models listed in the par/ directory for a given
        generation.

        Parameters
        ----------
        generation : int
            The generation to make a parameter table for.
        model : class
            This should be a class that has a run method, which given a
            parameter file, an output model directory, and a model name, will
            compute the model for that input.
        """

        print "[genetic] Generation %i: computing models" % generation

        # Loop over parameter files, and run each one in turn
        for par_file in glob.glob(os.path.join(self._parameter_dir(generation), '*.par')):
            model_name = string.split(os.path.basename(par_file), '.')[0]
            model.run(par_file, self._model_dir(generation), model_name)

    def compute_fits(self, generation, fitter):
        """
        For the generation specified, will compute the fit of all the models.

        The fitter argument should be used to pass a function that given a
        directory containing all the models, an output file, and a directory
        that can be used for plots, will output a table containing at least
        two columns named 'model_name' and 'chi2'.

        Parameters
        ----------
        generation : int
            The generation to make a parameter table for.
        fitter : class
            This should be a class that has a run method, which given a model
            directory, a fitting results file, and a plots directory, will run
            the fitting.

        """
        print "[genetic] Generation %i: fitting and plotting" % generation

        fitter.run(self._model_dir(generation),
                   self._fitting_results_file(generation),
                   self._plots_dir(generation))

    # The following are just convenience methods to return specific filenames
    # and paths for a given generation/model name

    def _generation_dir(self, generation):
        return self._models_dir + '/g%05i/' % generation

    def _parameter_file(self, generation, model_name):
        return self._parameter_dir(generation) + str(model_name) + '.par'

    def _model_prefix(self, generation, model_name):
        return self._model_dir(generation) + str(model_name)

    def _parameter_table(self, generation):
        return self._generation_dir(generation) + 'parameters.fits'

    def _fitting_results_file(self, generation):
        return self._generation_dir(generation) + 'fitting_output.fits'

    def _log_file(self, generation):
        return self._generation_dir(generation) + 'parameters.log'

    def _sampling_plot_file(self, generation):
        return self._generation_dir(generation) + 'sampling.eps'

    def _model_dir(self, generation):
        return self._generation_dir(generation) + 'models/'

    def _parameter_dir(self, generation):
        return self._generation_dir(generation) + 'par/'

    def _plots_dir(self, generation):
        return self._generation_dir(generation) + 'plots/'
