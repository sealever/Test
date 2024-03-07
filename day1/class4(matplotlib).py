import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2,2,1)
data = np.random.randint(0,2500,size=(50,50))
plt.imshow(data,cmap='Greys',alpha=0.5)
plt.subplot(2,2,2)
x = np.linspace(-5,5,50)
y = np.random.normal(-5,5,50)
plt.scatter(x,y,s=20,c='yellow',marker='D',edgecolors='red')
plt.subplot(2,2,3)
x = [1,2,3,4,5]
y = np.random.randint(1,5,size=5)
plt.bar(x,y,width=0.4,color='green')
for i,j in zip(x,y):
    plt.text(i,j+0.2,j,ha='center',va='center')
plt.subplot(2,2,4)
x = np.linspace(-5,5,30)
y=[]
for i in x**2:
    y.append(i)
    plt.plot(y,color='gold')
    plt.pause(0.2)
    # plt.clf()
plt.text(15,20,'y=x**2',ha='center',va='center')
plt.savefig('G:\AI-study\class\dir1\dir2\img.jpg')
plt.show()

