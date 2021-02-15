# -*- coding: utf-8 -*-
import numpy as np


def Softmax(x, axis=-1):
    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-16)
