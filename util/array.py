import numpy as nd
from functools import wraps


def arg_as_array(i=0):
    """ Converts the i-th argument to a numpy array.

    :param i: The index of the argument. By default, the first argument (0 index).
    :return: The modified function.
    """

    def _to_array(fun):
        def call(*args, **kwargs):
            res = list(args)
            res[i] = nd.asarray(args[i])
            res = tuple(res)
            return fun(*res, **kwargs)

        return call

    return _to_array
