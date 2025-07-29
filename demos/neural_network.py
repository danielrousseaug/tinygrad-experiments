#!/usr/bin/env python3
"""
Neural Network Demo for Tinygrad:
Trains a simple two-layer neural network on synthetic classification data.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Generate synthetic classification data (two classes)
    np.random.seed(0)
    num_samples = 200
    X_class0 = np.random.randn(num_samples, 2) + np.array([2, 2])
    X_class1 = np.random.randn(num_samples, 2) + np.array([-2, -2])
    X = np.vstack([X_class0, X_class1]).astype(np.float32)
    y = np.vstack([np.zeros((num_samples, 1)), np.ones((num_samples, 1))]).astype(np.float32)

    # Convert to tensors
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Initialize network parameters
    W1 = Tensor(np.random.randn(2, 16).astype(np.float32) * np.sqrt(2/2), requires_grad=True)
    b1 = Tensor(np.zeros(16, dtype=np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(16, 1).astype(np.float32) * np.sqrt(2/16), requires_grad=True)
    b2 = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    lr = 0.1
    epochs = 100

    logger.info("Starting training: lr=%s, epochs=%s", lr, epochs)
    for epoch in range(epochs):
        # Forward pass
        z1 = X_tensor @ W1 + b1
        a1 = z1.relu()
        z2 = a1 @ W2 + b2
        preds = z2.sigmoid()

        # Binary cross-entropy loss
        loss = -(y_tensor * preds.log() + (1 - y_tensor) * (1 - preds).log()).sum() / X.shape[0]

        # Backward pass
        for param in (W1, b1, W2, b2):
            param.grad = None
        loss.backward()

        # Update parameters
        for param in (W1, b1, W2, b2):
            param.data -= lr * param.grad

        # Log every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d: loss=%.6f", epoch + 1, loss.data)

    # Compute final accuracy
    final_preds = (preds.data > 0.5).astype(np.float32)
    accuracy = (final_preds == y).mean()

    logger.info("Training complete")
    logger.info("Final loss: %.6f, Accuracy: %.2f%%", loss.data, accuracy * 100)

if __name__ == "__main__":
    main()
