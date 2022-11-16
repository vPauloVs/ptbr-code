# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:08:27 2022

@author: paulo
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import r2_score


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


def divider(x, y, nordens, lambda0, razao):
    d = []  # comprimento medido
    step = []  # tamanho do passo

    n = len(y)
    xordens = []
    yordens = []
    for k in range(
        nordens
    ):  # repetição para preencher a lista step e comprimento medido (d)
        step.append(lambda0 / razao**k)  # variação do maior --> menor
        soma_dist = 0
        seca = [x[0], y[0]]
        fite = []
        xordem = [x[0]]
        yordem = [y[0]]

        for i in range(n):  # percorrer a função

            dist = dist_pontos(seca[0], seca[1], x[i], y[i])
            if dist >= step[k]:
                xordem.append(x[i])
                yordem.append(y[i])
                fite = [x[i], y[i]]
                dist = dist_pontos(seca[0], seca[1], fite[0], fite[1])
                soma_dist = soma_dist + dist
                seca = [x[i], y[i]]
        dist = dist_pontos(seca[0], seca[1], x[i], y[i])
        xordens.append(xordem)
        yordens.append(yordem)

        soma_dist = soma_dist + dist

        d.append(soma_dist)

    return step, d, xordens, yordens


def dist_pontos(x1, y1, x2, y2):
    dist = (
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    ) ** 0.5
    return dist


def apply_log(var):
    return np.log(var)


def fit_poly(x, y, order=1):
    a, b = np.polyfit(x, y, order)
    y_pred = a * x + b

    return a, b, y_pred


def r2(y, y_pred):
    return r2_score(y, y_pred)


def i_relativa(x, y, t, a):
    y_relativo = []
    x_relativo = []
    y_reta = []
    for k in range(t - 1):
        y_relativo.append((y[k + 1] - y[k]) / (x[k + 1] - x[k]))
        x_relativo.append(x[k])
        y_reta.append(a)

    return x_relativo, y_relativo, y_reta


def step_size(x, y):
    n = len(y)
    size = dist_pontos(x[0], y[0], x[n-1], y[n-1])
    return size


def graph_1(xordens, yordens, nordens):
    for i in range(nordens):
        plt.plot(xordens[i], yordens[i], label=i, lw=0.5)
    plt.legend()
    plt.title('Divisores da Função de Weierstrass')
    plt.show()


def graph_2(x_log, y_log, weierstrass_D):
    a, b, y_pred = fit_poly(x_log, y_log)
    plt.plot(x_log, y_log, "o", markersize=5)
    plt.grid()
    plt.xlabel('log(Tamanho do Passo)')
    plt.ylabel('log(Comprimento Total)')
    plt.legend()
    plt.title(f'Método dos Divisores 1D\nWeierstrass: D = {weierstrass_D}')
    plt.plot(x_log, y_pred,
             label=f'Inclinação = {round(a, 3)}')
    plt.legend()
    plt.show()


def graph_3(dx, dy, y_reta):
    plt.grid()
    plt.title('Métodos dos Divisores 1D\nGráfico das Variações')
    plt.plot(dx, dy, 'o', label='dy/dx')
    plt.plot(dx, y_reta, '-', label='Inclinação da Reta')
    plt.xlabel('log(Tamanho do Passo)')
    plt.ylabel('dy/dx')
    plt.legend()


def dados_entrada(weierstrass_D):
    x, y = weierstrass(weierstrass_D)
    return x, y


def parametros(size):
    lambda0 = size/50
    razao = 1.02
    nordens = 80
    return lambda0, razao, nordens


def dados_saida(x, y):
    size = step_size(x, y)
    lambda0, razao, nordens = parametros(size)
    step, d, xordens, yordens = divider(x, y, nordens, lambda0, razao)
    x_log = apply_log(step)  # log do tamanho do passo
    y_log = apply_log(d)  # log do comprimento medido
    return x_log, y_log, xordens, yordens, nordens


def det_dimensao(x_log, y_log, order=1):
    a, b, y_pred = fit_poly(x_log, y_log, order)
    divider_D = 1-a
    return divider_D, y_pred, a


def main_graficos(x, y, x_log, y_log, xordens, yordens,
                  nordens, weierstrass_D, a):
    # graph_1(xordens, yordens, nordens)
    graph_2(x_log, y_log, weierstrass_D)
    dx = []
    dy = []
    y_reta = []
    dx, dy, y_reta = i_relativa(x_log, y_log, nordens, a)
    graph_3(dx, dy, y_reta)


def main():
    weierstrass_D = 1.5
    x, y = dados_entrada(weierstrass_D)
    x_log, y_log, xordens, yordens, nordens = dados_saida(x, y)
    divider_D, y_pred, a = det_dimensao(x_log, y_log)
    main_graficos(x, y, x_log, y_log, xordens,
                  yordens, nordens, weierstrass_D, a)
    r2_val = r2(x_log, y_pred)
    return divider_D, r2_val


main()
