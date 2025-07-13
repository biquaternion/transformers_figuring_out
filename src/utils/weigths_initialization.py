#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def xavier_uniform(shape, random_state=42):
    rng = np.random.RandomState(random_state)
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return rng.uniform(low=-limit, high=limit, size=shape)


if __name__ == '__main__':
    pass