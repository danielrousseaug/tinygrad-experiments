#!/usr/bin/env python3
"""
XOR Demo for Tinygrad:
Trains a small 2-layer network to solve the XOR problem.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Data: XOR inputs and labels
    np.random.seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    X_t = Tensor(X)
    y_t = Tensor(y)

    # Initialize network parameters
    W1 = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    b1 = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(4, 1).astype(np.float32), requires_grad=True)
    b2 = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    lr = 0.1
    epochs = 2000

    logger.info("Starting XOR training: lr=%s, epochs=%s", lr, epochs)
    for epoch in range(epochs):
        # Forward pass
        z1 = X_t @ W1 + b1
        a1 = z1.relu()
        z2 = a1 @ W2 + b2
        preds = z2.sigmoid()

        # Binary cross-entropy loss
        loss = -(y_t * preds.log() + (1 - y_t) * (1 - preds).log()).sum() / X.shape[0]

        # Backward pass
        for param in (W1, b1, W2, b2):
            param.grad = None
        loss.backward()

        # Update parameters
        for param in (W1, b1, W2, b2):
            param.data -= lr * param.grad

        if (epoch + 1) % 500 == 0:
            logger.info("Epoch %d: loss=%.6f", epoch + 1, loss.data)

    # Final predictions
    final_preds = (((X_t @ W1 + b1).relu() @ W2 + b2).sigmoid().data > 0.5).astype(np.int32)
    logger.info("Final predictions: %s", final_preds.flatten())
    logger.info("Ground truth: %s", y.flatten().astype(np.int32))
    accuracy = (final_preds.flatten() == y.flatten().astype(np.int32)).mean()
    logger.info("Accuracy: %.2f%%", accuracy * 100)
    logger.info("XOR demo complete")

if __name__ == "__main__":
    main()
