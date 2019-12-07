"""
SOURCE: https://gist.github.com/alexalemi/2151722
"""

import math
import numpy as np

class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at: 
        http://www.johndcook.com/standard_deviation.html

    can take single values or iterables

    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean

    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M)*1./self.k
        newS = self.S + (x - self.M)*(x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std/math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S/(self.k-1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)

if __name__ == "__main__":
  w = Welford()
  arr = np.random.random(100)
  w.consume(arr)
  print(w.mean, w.std)
  print(arr.mean(), arr.std())