#!/usr/bin/env python3
"""
Autograd Demo for Tinygrad:
Demonstrates basic automatic differentiation on a simple computation.
"""
import logging

from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating tensors x and y with requires_grad=True")
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    logger.info("Computing z = (x + y) * y")
    z = (x + y) * y
    logger.info("z data: %s", z.data)

    logger.info("Performing backprop to compute gradients")
    z.backward()

    logger.info("Gradient dz/dx: %s", x.grad)
    logger.info("Gradient dz/dy: %s", y.grad)

if __name__ == "__main__":
    main()
