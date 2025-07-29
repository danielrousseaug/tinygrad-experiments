#!/usr/bin/env python3
"""
Sine Regression Demo for Tinygrad:
Fits a sine wave using a small neural network.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Generate data: y = sin(x) with noise
    np.random.seed(0)
    x = np.linspace(0, 2 * np.pi, 100, dtype=np.float32).reshape(-1, 1)
    y = np.sin(x)

    # Convert to tensors
    x_tensor = Tensor(x)
    y_tensor = Tensor(y)

    # Initialize network parameters
    W1 = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=True)
    b1 = Tensor(np.zeros(32, dtype=np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(32, 1).astype(np.float32), requires_grad=True)
    b2 = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    lr = 0.01
    epochs = 1000

    logger.info("Starting sine regression: lr=%s, epochs=%s", lr, epochs)
    for epoch in range(epochs):
        # Forward pass
        z1 = x_tensor @ W1 + b1
        a1 = z1.relu()
        preds = a1 @ W2 + b2

        # Mean squared error loss
        error = preds - y_tensor
        loss = (error * error).sum() / x.shape[0]

        # Backward pass
        for param in (W1, b1, W2, b2):
            param.grad = None
        loss.backward()

        # Update parameters
        for param in (W1, b1, W2, b2):
            param.data -= lr * param.grad

        if (epoch + 1) % 100 == 0:
            logger.info("Epoch %d: loss=%.6f", epoch + 1, loss.data)

    # Show a few predictions
    final_preds = (a1 @ W2 + b2).data
    logger.info("Final sample predictions vs target values:")
    for i in range(5):
        logger.info("x=%.2f, target=%.4f, pred=%.4f", x[i, 0], y[i, 0], final_preds[i, 0])

    logger.info("Sine regression complete")

if __name__ == "__main__":
    main()
