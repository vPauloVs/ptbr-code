# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:18:33 2022

@author: paulo
"""

import matplotlib.pyplot as plt


def regrecao(x, y):  # função que faz a regreção e plota a reta aproximada

    n = len(y)
    x_aprox = []
    y_aprox = []

    x_medio = 0
    y_medio = 0
    diferenca_x = []
    diferenca_y = []
    quadrado_difenreca_x = []
    multiplicacao_difenrenca = []
    for i in range(n):
        x_medio = x_medio + x[i]
        y_medio = y_medio + y[i]

    x_medio = x_medio/n
    y_medio = y_medio/n

    for i in range(n):
        diferenca_x.append(x[i]-x_medio)
        diferenca_y.append(y[i]-y_medio)
        quadrado_difenreca_x.append(diferenca_x[i]**2)
        multiplicacao_difenrenca.append(diferenca_x[i]*diferenca_y[i])

    coef_angular = sum(multiplicacao_difenrenca) / sum(quadrado_difenreca_x)
    coef_linear = y_medio - (coef_angular*x_medio)
    print('Coeficiente Angular - Função:', coef_angular)
    print('Coeficiente Linear - Função:', coef_linear)
    x_inicial = 0
    var_x = 1
    for i in range(n):
        x_aprox.append(x_inicial+(i*var_x))
        y_aprox.append(coef_linear + x_aprox[i]*coef_angular)

    plt.plot(x_aprox, y_aprox, '-')
    plt.title('Reta de Regressão - Pela Função')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return coef_linear, coef_angular


x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 2, 3, 4, 5]

lin, ang = regrecao(x, y)
