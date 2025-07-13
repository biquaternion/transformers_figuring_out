#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

    def step(self, lr):
        pass

    def parameters(self):
        return []


if __name__ == '__main__':
    pass