import os
from re import I
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


f = lambda x1,x2:(x1**2 + x2**2) 

x = np.array([[100],[200]])


def differentiation(function, x = np.ndarray):

    first_val  = tf.Variable(x[0], dtype=tf.float32)
    second_val = tf.Variable(x[1], dtype=tf.float32)

    with tf.GradientTape(watch_accessed_variables=True) as tape:
        y = function(first_val,second_val)

    df = tape.gradient(y,[first_val, second_val])
    ret = np.array([df[0], df[1]])
    return ret 

def optimization(f , x = np.ndarray, lerning_rate = 0.01, gamma = 0.0001):
    value = np.array([f(x[0], x[1])])
    arr = x
   
    i = 0
   
    while True:
    
      
        if(i==0 or i==1):
            x = x - ( lerning_rate*differentiation(f, x))
        else:
            x = x - ( lerning_rate*differentiation(f, x) + gamma * differentiation(f,arr[:,i-2]))

        
        arr   = np.append(arr, x, axis = 1)
        
        value = np.append(value, f(x[0], x[1]))

        if np.linalg.norm(differentiation(f, x))<0.001:
            break

        i = i+1
        print(x)
    return value, arr


value, arr = optimization(f ,x)


fig = plt.figure()
ax_3d = fig.add_subplot(projection = '3d')

X, Y = np.meshgrid(arr[0], arr[1])

ax_3d.plot_surface(X,Y,f (X,Y), color='r', rstride = 5, cmap = 'plasma')
ax_3d.plot(arr[0],arr[1],value, color = 'k')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')
plt.show()


