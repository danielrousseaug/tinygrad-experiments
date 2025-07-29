#!/usr/bin/env python3
"""
Mandelbrot ASCII Demo for Tinygrad:
Computes and prints an ASCII Mandelbrot fractal using tensor operations.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    width, height = 80, 24
    max_iter = 20

    logger.info("Generating grid of complex points (width=%d, height=%d)", width, height)
    re = np.linspace(-2.0, 1.0, width, dtype=np.float32)
    im = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    c_re = Tensor(re).reshape(1, width).repeat(height, 1)
    c_im = Tensor(im).reshape(height, 1).repeat(1, width)

    z_re = Tensor(np.zeros((height, width), dtype=np.float32))
    z_im = Tensor(np.zeros((height, width), dtype=np.float32))
    output = np.zeros((height, width), dtype=np.int32)

    logger.info("Iterating Mandelbrot with max iterations %d", max_iter)
    for i in range(max_iter):
        z_re2 = z_re * z_re
        z_im2 = z_im * z_im
        two_z_re_z_im = (z_re * z_im) * Tensor(2.0)
        z_re = z_re2 - z_im2 + c_re
        z_im = two_z_re_z_im + c_im
        mag2 = z_re * z_re + z_im * z_im

        mask = mag2.data > 4.0
        newly_escaped = (output == 0) & mask
        output[newly_escaped] = i

    logger.info("Rendering ASCII output")
    chars = " .:-=+*#%@"
    for row in output:
        line = "".join(chars[val * len(chars) // max_iter] for val in row)
        print(line)

    logger.info("Mandelbrot ASCII demo complete")

if __name__ == "__main__":
    main()
