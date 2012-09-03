import os
import sys

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
