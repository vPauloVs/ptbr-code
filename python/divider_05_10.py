# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import r2_score


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox))
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label = f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim=1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


def weierstrass(D = 1.1, f = 1.5, dom = 10, domit = 12, m = 100, T0 = 100., W0 = 0.):
    n=2 ** domit
    dt=1.*dom / n
    W=[]
    t=[]
    for k in range(n+1):
        t.append(T0+k * dt)
        Wf=0.0
        for j in range(0, m):
            Wf += math.sin(((f**j)*t[k]))/(f**(j*(2-D)))
        W.append(Wf)
    Wn=np.array(W)
    Wn=Wn-W0
    W=np.ndarray.tolist(Wn)
    return t, W


D=1.5
x, y=weierstrass(D)
plt.show()


n=len(y)
size=((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0=size/10  # (máxima distância - diametro do conjunto)
razao=1.2

nordens=10

d=[]  # comprimento medido
step=[]  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
    soma_dist=0
    seca=[x[0], y[0]]
    fite=[]
    xordem=[x[0]]
    yordem=[y[0]]
    n=len(y)
    for i in range(n):  # percorrer a função

        dist=(
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens)
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
    y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 06:00:09 2022

@author: paulo
"""


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


D = 1.5
x, y = weierstrass(D)
plt.show()


n = len(y)
size = ((x[n-1] - x[0])**2 + (y[n-1] - y[0])**2)**0.5

lambda0 = size/10  # (máxima distância - diametro do conjunto)
razao = 1.2

nordens = 10

d = []  # comprimento medido
step = []  # tamanho do passo

for k in range(nordens):  # repetição para preencher a lista step e comprimento medido (d)
    step.append(lambda0/razao**k)  # variação do maior --> menor
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


stepLog = np.log(step)  # x
dLog = np.log(d)  # y


plt.plot(stepLog, dLog, "o")
plt.grid()
plt.xlabel('log(Tamanho do Passo)')
plt.ylabel('log(Comprimento Total)')
plt.legend()
# plt.show()

#######################################################################
coeficiente_angular, coeficiente_linear = np.polyfit(stepLog, dLog, 1)


y_aprox = coeficiente_angular*stepLog + coeficiente_linear

y_derivada = []
for k in range(nordens):
y_derivada[k] = (dLog[k+1]-dLog[k-1]) / (stepLog[k+1]-stepLog[k])


r2 = r2_score(stepLog, y_aprox)
print(r2)

plt.title(f'Método dos Divisores 1D\nD = {D}')
plt.plot(stepLog, y_aprox,
         label=f'Inclinação = {round(coeficiente_angular, 3)}')
# f-string

plt.legend()
plt.show()


dim = 1 - coeficiente_angular
print('Dimensão Fractal:', round(dim, 3))
