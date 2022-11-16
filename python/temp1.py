# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
x=[]
y=[]
n=1000
d=[] #comprimento medido
L=10.
dx=L/n
seca=[0,0] #uma ponta do compasso
fite=[] #outra pornta do compasso
step=[0.1] #tamanho do passo
n=0 #conta passos

for i in range (n+1):
    x.append(i*dx)
    y.append(x[i]*x[i])
    #d = d+ ( ( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5 )

for k in range (10):   #repetição para preencher a lista step e comprimento medido (d)
    d[k]=0
    if k!=0:
        step[k]= step[k-1]+0.1  #variação no tamnho do passo
    for i in range (n+1): #percorrer a função
    
        if ( ( (x[i]-seca[0])**2 + (y[i]-seca[1])**2)**0.5 )>= step[k]:
            fite=[x[i], y[i]]
            d[k] = d[k] + (( (fite[0]-seca[0])**2 + (fite[1]-seca[1])**2)**0.5)
            seca =[fite[0], fite[1]]
            n = n+1
            
stepLog = []
dLog = []
for k in range (10): 
    stepLog[k] = np.log(step[k])
    dLog[k] = np.log(d[k])
    
    
plt.plot(x,y,'o')
#plt.plot(stepLog,dLog,'o')



        
