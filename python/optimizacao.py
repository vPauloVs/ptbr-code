# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:02:41 2022

@author: paulo
"""
import matplotlib.pyplot as plt
import numpy as np


def i_relativa(x, y, t):
    y_relativo = []
    x_relativo = []
    for k in range(t - 1):
        y_relativo.append((y[k + 1] - y[k]) / (x[k + 1] - x[k]))
        x_relativo.append(x[k])

    return x_relativo, y_relativo


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
y = [7, 7, 7.2, 7.1, 6.9, 6.7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1]

plt.plot(x, y, 'o')
plt.title('Função Genérica')
plt.xlabel('Eixo x')
plt.ylabel('Eixo y')
plt.show()

n = len(y)
x_relativo, y_relativo = i_relativa(x, y, n)

y_interesse = [[], ]
x_interesse = [[], ]

plt.plot(x_relativo, y_relativo, 'o')
plt.title('Iniclinações Relativas')
plt.show()
k = 0
c = 0
t = len(y_relativo)
for i in range(t-1):
    if(((y_relativo[i] < (1.2*y_relativo[i+1]))
       and (y_relativo[i] > (0.8*y_relativo[i+1])))
            or y_relativo[i] == (y_relativo[i+1])):

        y_interesse[k].append(y[i])
        x_interesse[k].append(x[i])
    #     c = + 1

    # else:
    #     if(c > 1):
    #         k += 1
    #         c = 0
    #         print("PASSANDO")


a = []
b = []
y_pred = []

a, b = np.polyfit(x, y, 1)
n = len(y)
for i in range(n):
    y_pred.append(a * x[i] + b)

plt.plot(x, y_pred, label='Regressão')
plt.plot(x_interesse[0], y_interesse[0], label='Retas de Interesse')
plt.plot(x, y, 'o', label='Dados')
plt.title('Problemas Optimização')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
plt.legend()
plt.show()
