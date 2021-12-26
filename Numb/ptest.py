
import numpy as np
from numba import jit
from time import process_time
import matplotlib.pyplot as plt

df = lambda x: np.array([2*x[0],2*x[1]])
f  = lambda x: x[0]**2 + x[1]**2

x = np.array([[100],[100]])




@jit(nopython = True)
def fun(x):
    return x[0]**2 + x[1]**2



def Optimix(x:np.ndarray, eps:float = 0.0001, step:float = 0.1)->np.ndarray:
    arr = np.array(x)
    val = np.array([fun(x)])
    
    while True:
        val = np.append(val, fun(x))
        x = x - step * df(x)
        arr = np.append(arr,x, axis = 1)
        if(np.linalg.norm(df(x))<eps):
            break
    return arr, val

def main():
    return Optimix(x)

t = process_time()

x, y = main()

print(process_time() - t)

fig = plt.figure()
ax_3d = fig.add_subplot(projection = '3d')

ax_3d.plot(x[0],x[1],y)
plt.show()
#print(y)
#print(z)