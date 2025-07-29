#!/usr/bin/env python3
"""
Softmax Classification Demo for Tinygrad:
Trains a small network on synthetic 3-class data with softmax cross-entropy.
"""
import logging

import numpy as np
from tinygrad.tensor import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Synthetic 3-class data (100 samples per class)
    np.random.seed(0)
    num_samples = 100
    X0 = np.random.randn(num_samples, 2) + np.array([2, 2])
    X1 = np.random.randn(num_samples, 2) + np.array([-2, 2])
    X2 = np.random.randn(num_samples, 2) + np.array([0, -2])
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.hstack([np.zeros(num_samples), np.ones(num_samples), 2 * np.ones(num_samples)]).astype(np.int32)

    # One-hot encode labels
    y_onehot = np.zeros((X.shape[0], 3), dtype=np.float32)
    y_onehot[np.arange(X.shape[0]), y] = 1.0

    X_t = Tensor(X)
    y_t = Tensor(y_onehot)

    # Initialize network parameters
    W1 = Tensor(np.random.randn(2, 16).astype(np.float32), requires_grad=True)
    b1 = Tensor(np.zeros(16, dtype=np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(16, 3).astype(np.float32), requires_grad=True)
    b2 = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)

    lr = 0.1
    epochs = 200

    logger.info("Starting softmax classification: lr=%s, epochs=%s", lr, epochs)
    for epoch in range(epochs):
        # Forward pass
        z1 = X_t @ W1 + b1
        a1 = z1.relu()
        logits = a1 @ W2 + b2
        exp_logits = logits.exp()
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -(y_t * probs.log()).sum() / X.shape[0]

        # Backward pass
        for param in (W1, b1, W2, b2):
            param.grad = None
        loss.backward()

        # Update parameters
        for param in (W1, b1, W2, b2):
            param.data -= lr * param.grad

        if (epoch + 1) % 50 == 0:
            pred_labels = np.argmax(probs.data, axis=1)
            acc = (pred_labels == y).mean()
            logger.info("Epoch %d: loss=%.6f, accuracy=%.2f%%", epoch + 1, loss.data, acc * 100)

    # Final evaluation
    pred_labels = np.argmax(probs.data, axis=1)
    acc = (pred_labels == y).mean()
    logger.info("Final loss: %.6f, Accuracy: %.2f%%", loss.data, acc * 100)
    logger.info("Softmax classification demo complete")

if __name__ == "__main__":
    main()
