import unittest
import numpy as np
from fast_stl.smooth import kernel_smoothing


class TestInputs(unittest.TestCase):
    def test_numpy_float32(self):
        x = np.arange(0, 10, 1, dtype=np.float32)
        y = np.arange(0, 10, 1, dtype=np.float32)
        kernel_smoothing(x, y, q=6)

    def test_numpy_float64(self):
        x = np.arange(0, 10, 1, dtype=np.float64)
        y = np.arange(0, 10, 1, dtype=np.float64)
        kernel_smoothing(x, y, q=6)

    def test_python_lists(self):
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 3, 4, 5, 6]
        kernel_smoothing(x, y, q=6)


class TestUsage(unittest.TestCase):
    def test_wellknown(self):
        # Example from https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
        x = [0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084, 4.7448394, 5.1073781,
             6.5411662, 6.7216176, 7.2600583, 8.1335874, 9.1224379, 11.9296663, 12.3797674,
             13.2728619, 14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354, 18.7572812]
        y = [18.63654, 103.49646, 150.35391, 190.51031, 208.70115, 213.71135, 228.49353,
             233.55387, 234.55054, 223.89225, 227.68339, 223.91982, 168.01999, 164.95750,
             152.61107, 160.78742, 168.55567, 152.42658, 221.70702, 222.69040, 243.18828]
        kernel_smoothing(x, y, q=7)


if __name__ == "__main__":    
    unittest.main()
