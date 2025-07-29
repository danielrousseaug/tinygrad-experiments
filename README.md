# Tinygrad Demos

This project contains some demos using Tinygrad, a minimalist DL framework. Each demo script is self-contained and has verbose logging to illustrate what's happening at each step.

## Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

For running tests (optional):

```bash
pip install pytest
```

## Demos

- **Autograd Demo**: `demos/autograd_demo.py` - Demonstrates Tinygrad's automatic differentiation with a simple tensor computation.
- **Linear Regression Demo**: `demos/linear_regression.py` - Fits a linear model to synthetic data using gradient descent.
- **Neural Network Demo**: `demos/neural_network.py` - Trains a simple two-layer neural network on synthetic classification data.
- **Sine Regression Demo**: `demos/sine_regression.py` - Fits a sine wave using a small neural network.
- **Mandelbrot ASCII Demo**: `demos/mandelbrot_ascii.py` - Computes and prints an ASCII Mandelbrot fractal using tensor operations.
- **XOR Demo**: `demos/xor_demo.py` - Trains a small two-layer network to solve the XOR problem.
- **Softmax Classification Demo**: `demos/softmax_demo.py` - Trains a small network on synthetic 3-class data with softmax cross-entropy.

Run a demo with:

```bash
python demos/<demo_script>.py
```

## Testing

To verify that all demos run without errors:

```bash
pytest
```
