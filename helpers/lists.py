import functools
import operator

def flatten(a):
    return functools.reduce(operator.concat, a)
