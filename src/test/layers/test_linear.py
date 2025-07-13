#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pytest
from src.layers.linear import Linear

@pytest.mark.parametrize("epochs, lr, tol", [(100, 0.1, 1e-2)])
def test_linear_learns_simple_mapping(epochs, lr, tol):
    np.random.seed(0)

    # y = 2 * x0 - 3 * x1 + 1
    in_features = 2
    out_features = 1
    X = np.random.randn(100, in_features)
    true_W = np.array([[2.0, -3.0]])
    true_b = 1.0
    y = X @ true_W.T + true_b

    layer = Linear(in_features, out_features)

    for _ in range(epochs):
        preds = layer.forward(X)
        # loss = np.mean((preds - y) ** 2)
        d_preds = 2 * (preds - y) / len(y)
        layer.backward(d_preds)
        layer.step(lr)

    w_err = np.abs(layer.weights.T - true_W.T)
    b_err = np.abs(layer.bias - true_b)

    assert np.all(w_err < tol), f"Weight error too large: {w_err}"
    assert np.all(b_err < tol), f"Bias error too large: {b_err}"


if __name__ == '__main__':
    pass