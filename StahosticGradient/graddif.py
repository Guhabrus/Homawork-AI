import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot  as plt

POINTS = 500


x = tf.random.uniform(shape = [POINTS], minval = 0, maxval = 10)
shum = tf.random.normal(shape = [POINTS], stddev=0.2)

k_tru = 0.7
b_tru = 2.0

y = k_tru*x+b_tru + shum


k = tf.Variable(2, dtype=tf.float32)
b = tf.Variable(2, dtype=tf.float32)

ls = lambda y_batch,f:tf.reduce_mean(tf.square(y_batch-f))

BATCH = 100
num_step = POINTS//BATCH

def Minimize(y, x, step = 0.01):
    
    VAL = np.array([])
    ARR = np.array([[],[]])
    for i in range(500):

        for n_batch in range(num_step):
            y_batch = y[n_batch*BATCH : (n_batch+1)*BATCH]
            x_batch = x[n_batch*BATCH : (n_batch+1)*BATCH]

            with tf.GradientTape() as type:
                f = k * x_batch + b
                loss = ls(y_batch, f)
        
            dk, db = type.gradient(loss, [k,b])
            k.assign_sub(step*dk)
            b.assign_sub(step*db)
        ARR = np.append(ARR, [[k],[b]], axis = 1)
        VAL = np.append(VAL,ls(y_batch, f))
        
    return k,b, VAL, ARR

k, b, v, arr = Minimize(y,x)
print("k = ",k,"\n", "b = ",b)
y1 = k*x+b
plt.scatter(x,y1, s = 3)
plt.scatter(x,k_tru*x+b_tru+shum, s = 3)
plt.show()




fig = plt.figure()
ax_3d = fig.add_subplot(projection = '3d')
X,Y = np.meshgrid(arr[0], arr[1])
XX,YY = np.meshgrid(x, v)



ax_3d.plot_surface(X,Y, YY , color = 'r', rstride = 5, cmap = 'plasma')
ax_3d.plot(arr[0], arr[1], v, color = 'k')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('y')
ax_3d.set_zlabel('z')
plt.show()