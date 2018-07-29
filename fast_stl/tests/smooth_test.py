import unittest
import numpy as np
from smooth import kernel_smoothing


class TestInputs(unittest.TestCase):
    def test_numpy_inputs(self):
        x = np.arange(0, 10, 1, dtype=np.float32)
        y = np.arange(0, 10, 1, dtype=np.float32)
        kernel_smoothing(x, y, q = 6)
        pass

    
if __name__ == "__main__":    
    unittest.main()