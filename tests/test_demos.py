"""
Tests for Tinygrad demo scripts.
"""
# ensure the parent repo's tinygrad package is importable
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import runpy


def test_autograd_demo():
    runpy.run_path('demos/autograd_demo.py', run_name='__main__')


def test_linear_regression():
    runpy.run_path('demos/linear_regression.py', run_name='__main__')


def test_neural_network():
    runpy.run_path('demos/neural_network.py', run_name='__main__')

def test_sine_regression():
    runpy.run_path('demos/sine_regression.py', run_name='__main__')

def test_mandelbrot_ascii():
    runpy.run_path('demos/mandelbrot_ascii.py', run_name='__main__')

def test_xor_demo():
    runpy.run_path('demos/xor_demo.py', run_name='__main__')

def test_softmax_demo():
    runpy.run_path('demos/softmax_demo.py', run_name='__main__')
