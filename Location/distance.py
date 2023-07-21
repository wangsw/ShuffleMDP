__author__ = 'westbrick'
# compute distances from points

import numpy as np


def norms(powers, roots, x, y):
    # compute roots(|x-y|^powers)
    diff = np.absolute(x-y)
    diff = np.power(diff, powers)
    diff = np.sum(diff)
    return np.power(diff, roots)


def l1norm(x, y):
    return norms(1, 1, x, y)


def l2norm(x, y):
    return norms(2, 0.5, x, y)


def sl2norm(x, y):
    return norms(2, 1, x, y)


def infnorm(x, y):
    return np.max(np.absolute(x-y))


def KLdivergence(x,y):
    return np.sum(x*np.log(y/x))


def todistances(points, norm):
    # compute distance oracle from points using norm
    ps = points.shape[0]
    ds = np.zeros((ps, ps))
    for i in range(0, ps):
        for j in range(0, ps):
            ds[i, j] = norm(points[i], points[j])
    return ds



