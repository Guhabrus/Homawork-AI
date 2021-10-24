import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot  as plt



fun = lambda x1,x2 : x1**2+x2**2
x = np.array([[100],[200]])


def differ(fun,x = np.ndarray):
    first_val  = tf.Variable(x[0], dtype=tf.float32)
    second_val = tf.Variable(x[1], dtype=tf.float32)

    with tf.GradientTape(watch_accessed_variables = True) as tape:
        y = fun(first_val,second_val)

    df  = tape.gradient(y, [first_val,second_val]) 
    ret = np.array([df[0], df[1]])
  
    return ret


def Optimix(fun, x = np.ndarray):
    value = np.array([fun(x[0],x[1])])
    arr = np.array(x)
    while True:
        x = x-0.01*differ(fun,x)
        arr = np.append(arr,x, axis=1)
        value = np.append(value, fun(x[1],x[0]))
    
        if np.linalg.norm(differ(fun,x))<0.001:
            break
    return value, arr    

value, arr = Optimix(fun,x)


fig = plt.figure()
ax_3d = fig.add_subplot(projection = '3d')

X, Y = np.meshgrid(arr[0], arr[1])

ax_3d.plot_surface(X,Y,fun(X,Y), color='r', rstride = 5, cmap = 'plasma')
ax_3d.plot(arr[0],arr[1],value, color = 'k')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')
plt.show()