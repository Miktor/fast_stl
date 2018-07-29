import numpy as np
import fast_stl.fast_stl_impl as impl

__all__ = ['kernel_smoothing']


def kernel_smoothing(x, y, target_x=None, q: int=None, f: float=None, weights=None):
    """
    :param x: Independent variable (should be the same length as y).
    :param y: Dependent variable (should be the same length as x). Missing data (NaNs) would be dropped.
    :param target_x: Points to sample the smoothed curve at (could be any length). If None, x would be used as target.
    :param q: "Size of the smoothing window". Window would be scaled to capture q samples from sample.
               Only one of q or f should be set.
    :param f: "Size of the smoothing window" from 0 to 1. 1 - means to take the whole series for smoothing.
               Only one of q or f should be set.
    :param weights: Additional weights which would be used in a local regression
    :return: Smoothed points at target_x positions
    """
    assert len(x) == len(y), 'Observations (x, y) should be the same length'
    assert (q is not None and f is None) or (q is None and f is not None), 'You should specify q or f, but not both'

    if target_x is None:
        target_x = x

    n = len(target_x)

    if f is not None:
        assert 0 < f, 'f should be greater than 0'
        q = int(np.ceil(f * n))

    if q is not None:
        assert q > 0, 'q must be greater than 0'

    return impl.loess(x, y, target_x, q)
