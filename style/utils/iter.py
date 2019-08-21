import functools
import operator


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)
