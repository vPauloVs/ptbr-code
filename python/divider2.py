# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:18:26 2022

@author: paulo
"""

import matplotlib.pyplot as plt
import numpy as np
x=[]
y=[]
n=1000
step=[] #tamanho do passo
L=10.
dx=L/n

fite=[] #uma ponta do compasso
seca=[0,0] #outra pornta do compasso
d=[]#comprimento medido
dis=0
tam= 0.
for i in range (n+1):  #gera o gráfico da função artificial para aplicação do metodo
    x.append(i*dx)
    y.append(x[i]*x[i])
plt.plot(x,y,'o')
plt.show()

for k in range (50): #repetição para preencher a lista step e comprimento medido (d)
    tam= tam + 0.1
    step.append(tam)
    soma_dist=0
    for i in range (n+1): #percorrer a função
    
        dist = ( ( (x[i]-seca[0])**2 + (y[i]-seca[1])**2)**0.5 )
        if dist >= step[k]:
            
            fite=[x[i], y[i]]
            soma_dist = soma_dist + (( (fite[0]-seca[0])**2 + (fite[1]-seca[1])**2)**0.5)
            seca =[x[i], y[i]]
            
     d.append(soma_dist)
stepLog = []
dLog = []

stepLog= np.log(step)
dLog= np.log(d)



plt.plot(stepLog,dLog,'o')
plt.show()