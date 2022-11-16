# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

x = np.arange (0,21,1)
y =[]
d=0
for i in range (21):
    y.append(x[i]*x[i])
    d = d+ ( ( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5 )

print(d)

plt.plot(x,y)