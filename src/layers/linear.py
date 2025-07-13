#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np

from src.utils.weigths_initialization import xavier_uniform

logger = logging.getLogger('Linear')
logging.basicConfig(level=logging.INFO)


class Linear:
    def __init__(self, in_features, out_features, random_state=42):
        logger.debug(f'Initializing Linear: {in_features}x{out_features}')
        self.weights = xavier_uniform((out_features, in_features))
        self.bias = np.random.randn(1, out_features)
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        self.x = None

    def forward(self, input):
        logger.debug(f'Forward: {input.shape}')
        self.x = input
        return input @ self.weights.T + self.bias

    def backward(self, d_output):
        self.dw = d_output.T @ self.x
        self.db = np.sum(d_output, axis=0, keepdims=True)
        dx = d_output @ self.weights
        return dx

    def step(self, lr):
        self.weights -= lr * self.dw
        self.bias -= lr * self.db


if __name__ == "__main__":
    in_features = 2
    out_features = 1
    lr = 0.1
    epochs = 100

    # y = 2 * x0 - 3 * x1 + 1
    np.random.seed(0)
    X = np.random.randn(100, in_features)  # [100, 2]
    true_W = np.array([[2.0], [-3.0]])  # [2, 1]
    true_b = 1.0
    y = X @ true_W + true_b  # [100, 1]

    layer = Linear(in_features, out_features)

    tol = 1e-8
    for epoch in range(epochs):
        preds = layer.forward(X)  # [100, 1]

        loss = np.mean((preds - y) ** 2)
        if loss < tol:
            logger.info(f'Epoch {epoch + 1}: loss {loss}')
            break
        d_preds = 2 * (preds - y) / len(y)  # dL/dy_pred
        layer.backward(d_preds)

        layer.step(lr)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:3d}: Loss = {loss:.4f}")

    logger.info(f'Trained weights:\t{layer.weights.T}\n'
                f'Trained bias:\t{layer.bias}'
                f'Expected:\t{true_W.T}'
                f'Expected bias:\t{true_b}')
