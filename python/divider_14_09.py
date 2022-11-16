# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:18:26 2022

@author: paulo
"""


import matplotlib.pyplot as plt
import numpy as np
import math


def weierstrass(D=1.1, f=1.5, dom=10, domit=12, m=100, T0=100., W0=0.):
    n = 2 ** domit
    dt = 1.*dom / n
    W = []
    t = []
    for k in range(n+1):
        t.append(T0+k * dt)
        Wf = 0.0
        for j in range(0, m):
            Wf += math.sin(((f**j)*t[k]))/(f**(j*(2-D)))
        W.append(Wf)
    Wn = np.array(W)
    Wn = Wn-W0
    W = np.ndarray.tolist(Wn)
    return t, W


D = 1.6
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2+(y[n-1] - y[0])**2)**0.5

lambda0 = size/30

razao = 1.05
nordens = 40


d = []  # comprimento total
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)
    soma_dist = 0
    seca = [x[0], y[0]]
    fite = []
    xordem = [x[0]]
    yordem = [y[0]]
    n = len(y)
    for i in range(n):  # percorrer a função

        dist = (
            (x[i] - seca[0]) ** 2 + (y[i] - seca[1]) ** 2
        ) ** 0.5  # compara a distancia da ponta seca com o ponto
        if dist >= step[k]:
            xordem.append(x[i])
            yordem.append(y[i])
            fite = [x[i], y[i]]
            soma_dist = soma_dist + (
                ((fite[0] - seca[0]) ** 2 + (fite[1] - seca[1]) ** 2) ** 0.5
            )
            seca = [x[i], y[i]]
    soma_dist = soma_dist + ((fite[0] - seca[0])
                             ** 2 + (fite[1] - seca[1]) ** 2) ** 0.5
    d.append(soma_dist)
    plt.plot(xordem, yordem, label=k)


plt.legend()
plt.title('Função de Weierstrass')
plt.show()

stepLog = []
dLog = []


# for i in range(nordens):
#     dLog.append(d[i])
#     stepLog.append(step[i])
#     if d[i]/d[i+1] <= ????
#         break


stepLog = np.log(step)
dLog = np.log(d)


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()


coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)
x_aprox = []
y_aprox = []
x_inicial = stepLog[-1]

var_x = 0.1
for i in range(n):
    x_aprox.append(x_inicial+var_x*i)
    y_aprox.append(coeficiente_linear + x_aprox[i]*coeficiente_angular)
    if(x_aprox[i] >= stepLog[0]):
        break


plt.title('Método dos Divisores 1D')
plt.plot(x_aprox, y_aprox, label='Inclinação = %s' %
         round(coeficiente_angular, 3))

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 2))
