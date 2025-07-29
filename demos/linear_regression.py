#!/usr/bin/env python3
"""
Linear Regression Demo for Tinygrad:
Fits a line y = 3x + 2 with noise using gradient descent.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Create synthetic data: y = 3x + 2 + noise
    np.random.seed(0)
    xs = np.linspace(0, 1, 100, dtype=np.float32).reshape(-1, 1)
    ys = 3 * xs + 2 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    # Convert data to tensors
    x_tensor = Tensor(xs)
    y_tensor = Tensor(ys)

    # Initialize parameters w and b
    w = Tensor(np.random.randn(1, 1).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    lr = 0.1
    epochs = 100

    logger.info("Starting training: lr=%s, epochs=%s", lr, epochs)
    for epoch in range(epochs):
        # Forward pass: predictions and loss (MSE)
        preds = x_tensor @ w + b
        error = preds - y_tensor
        loss = (error * error).sum() / xs.shape[0]

        # Backward pass
        w.grad = None
        b.grad = None
        loss.backward()

        # Update parameters
        w.data -= lr * w.grad
        b.data -= lr * b.grad

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d: loss=%.6f", epoch + 1, loss.data)

    logger.info("Training complete")
    logger.info("Learned parameters: w=%.4f, b=%.4f", w.data.flatten()[0], b.data.flatten()[0])

if __name__ == "__main__":
    main()
