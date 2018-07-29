import numpy as np
import fast_loess

def test():
    fast_loess.example(np.arange(0, 10, 1, dtype=np.float32), np.arange(0, 10, 1, dtype=np.float32), np.arange(0, 6, 1, dtype=np.float32), 4)


if __name__ == "__main__":
    print('Running benchmarks')

    test()
