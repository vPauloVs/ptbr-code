# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

print ('Determinar os intervalos')
mini = input('Inicio:')
maxi = input('Fim:')

x = np.arange (mini,maxi,1)
y =[]
d=0
for i in range (mini+maxi):
    y.append(x[i]*x[i]*x[i])
    d = d+ ( ( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5 )
         

plt.plot(x,y)





        
