import numpy as np

# Modified from
# https://introcs.cs.princeton.edu/python/22module/gaussian.py.html

BOUNDED_INFINITY = 10.
def pdf(orig_x, mu=0.0, sigma=1.0):
    sigma = sigma.astype(np.float)
    orig_x = orig_x.astype(np.float)
    x = orig_x - mu / sigma
    a = np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma
    return np.where(np.logical_or(np.logical_or(np.isinf(a), np.isnan(a)), sigma == 0),
        np.where(orig_x == mu, np.ones_like(mu) * BOUNDED_INFINITY, np.zeros_like(mu)),
        a
    )
    # return np.divide(a, sigma, out=np.isclose(orig_x, mu).astype(np.float) * BOUNDED_INFINITY, where=sigma!=0)

def cdf(z, mu=0.0, sigma=1.0):
    z = z - mu / sigma
    if z < -8.0: return 0.0
    if z > +8.0: return 1.0
    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z * z / i
        i += 2
    return 0.5 + total * pdf(z)
