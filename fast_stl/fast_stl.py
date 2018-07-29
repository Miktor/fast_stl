import numpy as np
from fast_stl_impl import loess


def test():
    loess(np.arange(0, 10, 1, dtype=np.float32), np.arange(0, 10, 1, dtype=np.float32), np.arange(0, 6, 1, dtype=np.float32), 4)


if __name__ == "__main__":
    print('Running benchmarks')
    test()
