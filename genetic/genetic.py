import sys
import string
import glob
import os
import time
import random as r
import signal

import atpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mpl
import multiprocessing as mp
import subprocess

try:
    from mpi4py import MPI
    mpi_enabled = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    status = MPI.Status()
except:
    mpi_enabled = False
    rank = 0
    nproc = 1

wait_time = 1.
delta = 0.1
n_max_sample = 10000

# The following is Steven Bethard's functions to pickle methods - required to
# use multiprocessing.Pool with model.run

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# Define a class to allow multiple arguments to be passed to model.run

class RunWrapper(object):

    def __init__(self, function):
        self.function = function

    def __call__(self, args):
        return self.function(*args)


def low_cpu_barrier():

    if rank == 0:

        for dest in range(1, nproc):
            print "[mpi] rank 0 sending exit to rank %i" % dest
            comm.send({'model': 'exit'}, dest=dest, tag=3)

    else:

        while True:
            status = MPI.Status()
            comm.Iprobe(source=0, tag=3, status=status)
            if status.source == 0:
                break
            time.sleep(delta)

        data = comm.recv(source=0, tag=3)

    comm.barrier()


def kill_all(ppid):

    pids = os.popen("ps -x -e -o pid,ppid | awk '{if($2 == " + str(ppid) + ") print $1}'").read().split()

    if type(pids) != list:
        pids = (pids, )

    for pid in pids:
        pid = pid.strip()
        if pid:
            pid = int(pid)
            kill_all(pid)

    try:
        os.kill(ppid, signal.SIGKILL)
        print "PID: ", ppid, " killed"
    except:
        print "PID: ", ppid, " does not exist"

    return


def etime(pid):
    cols = os.popen('ps -xe -o pid,etime | grep ' + str(pid)).read().strip().split()
    if len(cols) == 0:
        return 0
    else:
        cols = cols[1].split(':')
        if len(cols) == 2:
            d = 0
            h = 0
            m, s = cols
        elif len(cols) == 3:
            h, m, s = cols
            if '-' in h:
                d, h = h.split('-')
            else:
                d = 0
        elif len(cols) == 4:
            d, h, m, s = cols
        else:
            raise Exception("Can't understand " + str(cols))

        d = float(d)
        h = float(h)
        m = float(m)
        s = float(s)

        return ((d * 24 + h) * 60 + m) * 60 + s


def kill_inactive(seconds):
    for p in mp.active_children():
        if etime(p.pid) > seconds:
            print "Process %i has exceeded %i seconds, terminating" % (p.pid, seconds)
            kill_all(p.pid)


def wait_with_timeout(p, seconds):
    while True:
        if not p in mp.active_children():
            break
        if etime(p.pid) > seconds:
            print "Process %i has exceeded %i seconds, terminating" % (p.pid, seconds)
            kill_all(p.pid)
        time.sleep(wait_time)


def create_dir(dir_name):
    delete_dir(dir_name)
    os.system("mkdir " + dir_name)


def delete_dir(dir_name):
    if os.path.exists(dir_name):
        reply = raw_input("Delete directory " + dir_name + "? [y/[n]] ")
        if reply == 'y':
            os.system('rm -r ' + dir_name)
        else:
            print "Aborting..."
            sys.exit()


# Tournament selection routine
# Good parameters for getting ~10% are k_frac=0.2 and p=0.9

def select(chi2, n, k_frac, p):

    k = int(len(chi2) * k_frac)

    assert k > 0, "k_frac is too small"

    model_id = [i for i in range(len(chi2))]

    prob = [p * (1 - p) ** j for j in range(k)]
    norm = sum(prob)
    for i in range(len(prob)):
        prob[i] = prob[i] / norm

    choices = []

    for t in range(n):

        pool_id = r.sample(model_id, k)

        pool_chi = chi2[pool_id]

        aux_list = zip(pool_chi, pool_id)
        aux_list.sort()
        pool_chi, pool_id = map(list, zip(*aux_list))

        xi = r.random()
        for j in range(k):
            if(xi <= sum(prob[0: j + 1])):  # is  + 1 because prob[0: 0] is empty
                choice = pool_id[j]
                choices.append(choice)
                break

    return(choices)


class Genetic(object):

    def __init__(self, n_models, output_dir, template, configuration,
                 existing=False, fraction_output=0.1, fraction_mutation=0.5,
                 mode='serial', n_cores=None, max_time=600):
        '''
        The Genetic class is used to control the SED fitter genetic algorithm

        Parameters
        ----------

        n_models: int
            Number of models to run in the first generation, and to keep in
            subsequent generations

        output_dir: str
            The directory in which to output all the models

        template: str
            The template parameter file

        configuration: str
            The configuration file that describes how the parameters should be
            sampled. This file should contain four columns:
                * The name of the parameter (no spaces)
                * Whether to sample linearly ('linear') or logarithmically
                  ('log')
                * The minimum value of the range
                * The maximum value of the range

        existing: bool, optional
           Whether to keep any existing model directory

        fraction_output: float, optional
           Fraction of models to add to and remove from the pool at each
           generation

        fraction_mutation: float, optional
           Fraction of children that are mutations (vs crossovers)

        mode: str, optional
            How to run the models. Can be one of 'serial' (one model at a
            time), 'multiprocessing' (using multiple cores on a single
            machine), or 'mpi' (using MPI on a computer cluster).

        n_cores: int, optional
           Number of cores that can be used to compute models if using the
           multiprocessing mode. If using MPI, then this option is ignored,
           and the number of cores is set by mpirun/mpiexec.

        max_time: float, optional
           Maximum number of seconds a model can run for
        '''

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
        self._max_time = max_time

        if mode in ['serial', 'serial_file']:
            self._mode = mode
            if n_cores is not None:
                raise Exception("Cannot set n_cores in serial mode")
        elif mode == 'mpi':
            if not mpi_enabled:
                raise Exception("Can't use MPI, mpi4py did not import correctly")
            self._mode = mode
            if n_cores is not None:
                raise Exception("Cannot set n_cores in mpi mode")
        elif mode == 'multiprocessing':
            self._mode = mode
            if n_cores is None:
                raise Exception("Need to set n_cores in multiprocessing mode")
            self._n_cores = n_cores
        else:
            raise Exception("mode should be one of serial/mpi/multiprocessing")

        # Create output directory
        if not existing and (not self._mode == 'mpi' or rank == 0):
            create_dir(self._models_dir)

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

    def initialize(self, generation):
        '''
        Initialize the directory structure for the generation specified.
        '''
        if not self._mode == 'mpi' or rank == 0:
            create_dir(self._generation_dir(generation))
            create_dir(self._model_dir(generation))
            create_dir(self._parameter_dir(generation))
            create_dir(self._plots_dir(generation))
        return

    def make_par_table(self, generation, validate=lambda x: True):
        '''
        Creates a table of models to compute for the generation specified.

        If this is the first generation, then the parameters are sampled in an
        unbiased way within the ranges specified by the user. Otherwise, this
        method uses results from previous generations to determine which
        models to run.
        '''
        if not self._mode == 'mpi' or rank == 0:

            print "[genetic] Generation %i: making parameter table" % generation

            t = atpy.Table()

            if generation == 1:

                print "Initializing parameter file for first generation"

                # Create model names column
                t.add_column('model_name', ["g1_" + str(i) for i in range(self.n_models)], dtype='|S30')

                # Create empty columns in table
                for par_name in self.parameters:
                    t.add_empty_column(par_name, dtype=float)

                for i in range(self.n_models):

                    for sample in range(n_max_sample):

                        for par_name in self.parameters:

                            mode = self.parameters[par_name]['mode']
                            vmin = self.parameters[par_name]['min']
                            vmax = self.parameters[par_name]['max']

                            if mode == "linear":
                                value = r.uniform(vmin, vmax)
                            elif mode == "log":
                                value = 10. ** r.uniform(np.log10(vmin), np.log10(vmax))
                            else:
                                raise Exception("Unknown mode: %s" % mode)

                            t.data[par_name][i] = value

                        if validate(t.data[i]):
                            break

                    if sample == n_max_sample - 1:
                        raise Exception("Could not sample a valid model after {:d} tries".format(n_max_sample))

            else:

                n_output = int(self.n_models * self._fraction_output)

                # Create model names column
                t.add_column('model_name', ["g%i_%i" % (generation, i) for i in range(n_output)], dtype='|S30')

                # Read in previous parameter tables

                par_table = atpy.Table(self._parameter_table(1),
                                       verbose=False)
                for g in range(2, generation):
                    par_table.append(atpy.Table(self._parameter_table(g),
                                     verbose=False))

                model_names = np.array([x.strip() for x in par_table.model_name.tolist()])

                for column in par_table.names:
                    if column != 'model_name':
                        t.add_empty_column(column,
                                           par_table.columns[column].dtype)

                # Read in fitter results, and sort from best to worst-fit chi^2

                chi2_table = atpy.Table(self._fitting_results_file(1),
                                        verbose=False)
                for g in range(2, generation):
                    chi2_table.append(atpy.Table(self._fitting_results_file(g), verbose=False))

                chi2_table.sort('chi2')

                order = np.argsort(chi2_table.chi2)

                # Truncate the table to the n_models first models

                # chi2_table = chi2_table.rows(range(self.n_models))
                chi2_table = chi2_table.rows(order[: self.n_models])

                print "Best fit so far: ", chi2_table.data[0]

                selected = []

                mutations = 0
                crossovers = 0

                logfile = file(self._log_file(generation), 'wb')

                for i in range(0, n_output):

                    # Select whether to do crossover or mutation

                    if(r.random() > self._fraction_mutation):

                        crossovers += 1

                        im1, im2 = select(chi2_table.chi2, n=2, k_frac=0.1, p=0.9)

                        m1 = chi2_table.model_name[im1]
                        m2 = chi2_table.model_name[im2]

                        logfile.write('g%s_%i = crossover of %s and %s\n' % (generation, i, m1, m2))

                        selected.append(im1)
                        selected.append(im2)

                        par_m1 = par_table.row(np.char.strip(par_table.model_name) == m1.strip())
                        par_m2 = par_table.row(np.char.strip(par_table.model_name) == m2.strip())

                        for sample in range(n_max_sample):

                            for par_name in par_table.names:

                                if par_name != 'model_name':

                                    par1 = par_m1[par_name]
                                    par2 = par_m2[par_name]

                                    xi = r.uniform(0., 1.)

                                    mode = self.parameters[par_name]['mode']

                                    if mode == "linear":
                                        value = par1 * xi + par2 * (1. - xi)
                                    elif mode == "log":
                                        value = 10. ** (np.log10(par1) * xi + np.log10(par2) * (1. - xi))
                                    else:
                                        raise Exception("Unknown mode: %s" % mode)

                                    t.data[par_name][i] = value

                            if validate(t.data[i]):
                                break

                        if sample == n_max_sample - 1:
                            raise Exception("Could not sample a valid model after {:d} tries".format(n_max_sample))

                    else:

                        mutations += 1

                        im1 = select(chi2_table.chi2, n=1, k_frac=0.1, p=0.9)[0]

                        m1 = chi2_table.model_name[im1]

                        logfile.write('g%s_%i = mutation of %s\n' % (generation, i, m1))

                        selected.append(im1)

                        mutation = r.choice(par_table.names)

                        par_m1 = par_table.row(np.char.strip(par_table.model_name) == m1.strip())

                        for sample in range(n_max_sample):

                            for par_name in par_table.names:

                                if par_name != 'model_name':

                                    value = par_m1[par_name]

                                    if par_name == mutation:

                                        mode = self.parameters[par_name]['mode']
                                        vmin = self.parameters[par_name]['min']
                                        vmax = self.parameters[par_name]['max']

                                        if mode == "linear":
                                            value = r.uniform(vmin, vmax)
                                        elif mode == "log":
                                            value = 10. ** r.uniform(np.log10(vmin), np.log10(vmax))
                                        else:
                                            raise Exception("Unknown mode: %s" % mode)

                                    t.data[par_name][i] = value

                            if validate(t.data[i]):
                                break

                        if sample == n_max_sample - 1:
                            raise Exception("Could not sample a valid model after {:d} tries".format(n_max_sample))

                logfile.close()

                print "   Mutations  : " + str(mutations)
                print "   Crossovers : " + str(crossovers)

                fig = mpl.figure()
                ax = fig.add_subplot(111)
                ax.hist(selected, 50)
                fig.savefig(self._sampling_plot_file(generation))

            t.write(self._parameter_table(generation), verbose=False)

        return

    def make_par_indiv(self, generation, parser, interpreter=None):
        '''
        For the generation specified, will read in the parameters.fits file
        and the parameter file template, and will output individual parameter
        files to the par/ directory for that given generation.

        The parser argument should be used to pass a function that given a
        line from the parameter file will return the parameter name.

        Optionally, one can specify an interpreting function that given a
        parameter name and a dictionary of parameter values, will determine
        the actual value to use (useful for example if several parameters are
        correlated).
        '''

        if not self._mode == 'mpi' or rank == 0:

            print "[genetic] Generation %i: making individual parameter files" % generation

            # Read in table and construct dictionary
            table = atpy.Table(self._parameter_table(generation))
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
                f = file(self._parameter_file(generation, model_name), 'wb')

                # Cycle through template lines
                for line in self._template:
                    name, value = parser(line)
                    if value == 'VAR':
                        if interpreter:
                            value = interpreter(generation, name, dict(model.items() + static.items()))
                        else:
                            value = model[name]
                        f.write(line.replace('VAR', str(value)))
                    else:
                        f.write(line)

                f.close()

        if self._mode == 'mpi':
            low_cpu_barrier()

        return

    def compute_models(self, generation, model):
        '''
        For the generation specified, will compute all the models listed in
        the par/ directory.

        The model argument should be used to pass a function that given a
        parameter file and an output model directory will compute the model
        for that input and produce output with the specified prefix.
        '''

        start_dir = os.path.abspath(".")

        if self._mode in ['serial', 'multiprocessing']:

            # Wrapper function to be able to use multiple arguments in map
            # def run_wrapper(args):
            #     return model.run(*args)

            run_wrapper = RunWrapper(model.run)

            # Define arguments
            models = []
            for par_file in glob.glob(os.path.join(self._parameter_dir(generation), '*.par')):
                model_name = string.split(os.path.basename(par_file), '.')[0]
                models.append((par_file, self._model_dir(generation), model_name))

            # Run the models using map
            if self._mode == 'serial':
                print "[genetic] Generation %i: computing models in serial mode" % generation
                map(run_wrapper, models)
            else:
                print "[genetic] Generation %i: computing models using multiprocessing" % generation
                p = mp.Pool(processes=self._n_cores)
                p.map(run_wrapper, models)

        elif ['serial_file']:

            # This mode allows the model running function to be called via the
            # command-line instead of via a direct function call. The reason
            # for wanting to do this is that this opens up many possibilities,
            # including that the file being called could e.g. submit a job to
            # a cluster. In this mode, we wait for the return code of the
            # script to indicate that the run is complete.

            if not isinstance(model, basestring):
                raise ValueError("model should be the path to a script")

            # Loop over models and start up the process for each model
            processes = []
            for par_file in glob.glob(os.path.join(self._parameter_dir(generation), '*.par')):
                model_name = string.split(os.path.basename(par_file), '.')[0]
                p = subprocess.Popen([model, par_file, self._model_dir(generation), model_name])
                processes.append(p)

            while True:

                # For some reason time.sleep(...) doesn't work here, so we have to do it the old-fashioned way
                import time
                time1 = time.time()
                while time.time() < time1 + 1.:
                    pass

                # Check whether we can exit
                status = [p.poll() for p in processes]
                if status.count(None) == 0:
                    print "[genetic] models done, exiting"
                    break
                else:
                    print "[genetic] %i models running" % status.count(None)

        else:

            low_cpu_barrier()

            if rank == 0:

                print "[genetic] Generation %i: computing models with %i processes (using MPI)" % (generation, nproc)

                for par_file in glob.glob(os.path.join(self._parameter_dir(generation), '*.par')):

                    print "[mpi] rank 0 waiting for communications"

                    while True:
                        status = MPI.Status()
                        comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status)
                        if status.source > 0:
                            break
                        time.sleep(delta)

                    data = comm.recv(source=status.source, tag=1)
                    if data['status'] == 'ready':
                        print "[mpi] rank 0 received ready from rank %i" % data['source']
                        print "[mpi] rank 0 sending model %s to rank %i" % (par_file, data['source'])
                        comm.send({'model': par_file}, dest=data['source'], tag=2)
                    else:
                        raise Exception("Got unexpected status: %s" % data['status'])

                stopped = np.zeros(nproc, dtype=bool)

                while True:

                    if np.all(stopped[1:]):
                        break

                    print "[mpi] rank 0 waiting for communications"

                    while True:
                        status = MPI.Status()
                        comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status)
                        if status.source > 0:
                            break
                        time.sleep(delta)

                    data = comm.recv(source=status.source, tag=1)
                    if data['status'] == 'ready':
                        print "[mpi] rank 0 received ready from rank %i" % data['source']
                        print "[mpi] rank 0 sending stop to rank %i" % data['source']
                        comm.send({'model': 'stop'}, dest=data['source'], tag=2)
                        stopped[data['source']] = True
                    else:
                        raise Exception("Got unexpected status: %s" % data['status'])

            else:

                while True:

                    print "[mpi] rank %i is ready" % rank
                    comm.send({'status': 'ready', 'source': rank}, dest=0, tag=1)
                    data = comm.recv(source=0, tag=2)

                    if data['model'] == 'stop':
                        print "[mpi] rank %i has finished running models" % rank
                        break

                    par_file = data['model']

                    print "[mpi] rank %i running model: %s" % (rank, par_file)

                    # Prepare model name and output filename
                    model_name = string.split(os.path.basename(par_file), '.')[0]

                    # Prepare thread
                    os.chdir(start_dir)
                    p = mp.Process(target=model.run, args=(par_file, self._model_dir(generation), model_name))
                    p.start()
                    wait_with_timeout(p, self._max_time)

        if self._mode == 'mpi':
            low_cpu_barrier()

        return

    def compute_fits(self, generation, fitter):
        '''
        For the generation specified, will compute the fit of all the models.

        The fitter argument should be used to pass a function that given a
        directory containing all the models, an output file, and a directory
        that can be used for plots, will output a table containing at least
        two columns named 'model_name' and 'chi2'.
        '''
        if not self._mode == 'mpi' or rank == 0:
            print "[genetic] Generation %i: fitting and plotting" % generation
            fitter.run(self._model_dir(generation), self._fitting_results_file(generation), self._plots_dir(generation))
        if self._mode == 'mpi':
            low_cpu_barrier()
        return
