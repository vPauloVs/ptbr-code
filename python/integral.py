# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:49:53 2022

@author: paulo
"""

import numpy as np
import math


def weierstrass(D=1.1, f=1.5, dom=10, domit=12, m=100, T0=100.0, W0=0.0):
    n = 2**domit
    dt = 1.0 * dom / n
    W = []
    t = []
    for k in range(n + 1):
        t.append(T0 + k * dt)
        Wf = 0.0
        for j in range(0, m):
            Wf += math.sin(((f**j) * t[k])) / (f ** (j * (2 - D)))
        W.append(Wf)
    Wn = np.array(W)
    Wn = Wn - W0
    W = np.ndarray.tolist(Wn)
    return t, W


D = 1.4

x, y = weierstrass(D)

n = len(y)

area = 0
for i in range(n-1):
    if(y[i] >= 0):
        area = area + ((x[i+1] - x[i]) * y[i])
    else:
        area = area + ((x[i+1] - x[i]) * y[i]*(-1))
    print(area)
