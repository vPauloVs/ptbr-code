# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:16:19 2022

@author: paulo
"""
import matplotlib.pyplot as plt
x=[]
y=[]
n=100
d=0 #comprimento medido
L=10.
dx=L/n
for i in range (n+1):
    x.append(i*dx)
    y.append(x[i]*x[i])

for i in range (1,n+1):
    d = d+ ( ( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5 )
    
plt.plot(x,y,'o')