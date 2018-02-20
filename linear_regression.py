import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt


learning_rate=0.0000001
iterations=55
excel_file='Gray Kangaroos.xls'

def init_weights():
    return np.random.uniform(-1,1,2)
def cost(x,y,theta):
    return np.sum(((predict(x,theta)-y)**2)/2*len(x))
def predict(x,theta):
    return theta[1] + x.dot(theta[0])

def update_weights(x, y, weight, bias, learning_rate):
    wd = 0
    bd = 0
    n = len(x)
    
    for i in range(n):
        
        wd += -2*x[i] * (y[i] - (weight*x[i] + bias))

        
        bd += -2*(y[i] - (weight*x[i] + bias))
        
    
    
    

    
    weight -= (wd / n) * learning_rate
    bias -= (bd / n) * learning_rate

    return weight, bias

def start():
    a=np.sort(np.asarray(pd.read_excel(excel_file)),axis=0)
    
    X=a[...,0]
    Y=a[...,1]


    plt.plot(X,Y,'ro')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


    theta=init_weights()
    costs=[]
    thetas=[]
    for i in (range(0,iterations)):
        thetas.append(theta)
        theta[0],theta[1]=update_weights(X,Y,theta[0],theta[1],learning_rate)
        costs.append(cost(X,Y,theta))
    plt.plot(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost_log")
    plt.show()   

    plt.plot(X,Y,'ro',X,predict(X,theta))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    start()
