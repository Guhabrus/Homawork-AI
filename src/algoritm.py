import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os


def show_all(_file = ""):
    """Функция для показа всей сцены
    file - путь\имя файла сцены формата .xyz"""
    

def data_process(_file = ""):

    """Функция кластеризации сцены\n 
    file - путь\имя файла сцены формата .xyй"""

    if(not os.path.exists(_file)):
        print("[DATA_PROCESS]--error--> Пошел в жопу, где файл????")
        return False

    x,y,z,illuminance,reflectance,intensity,nb_of_returns = np.loadtxt(_file,skiprows=1, delimiter=';', unpack=True)

    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.scatter(x, z, c=intensity, s=0.05)
    plt.axhline(y=np.mean(z), color='r', linestyle='-')
    plt.title("First view")
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')

    plt.subplot(1, 2, 2) # index 2
    plt.scatter(y, z, c=intensity, s=0.05)
    plt.axhline(y=np.mean(z), color='r', linestyle='-')
    plt.title("Second view")
    plt.xlabel('Y-axis')
    plt.ylabel('Z-axis')

    plt.show()


    pcd=np.column_stack((x,y,z))
    mask=z>np.mean(z)
    spatial_query=pcd[z>np.mean(z)]


    ax = plt.axes(projection='3d')
    ax.scatter(x[mask], y[mask], z[mask], c = intensity[mask], s=0.1)
    plt.show()

    plt.scatter(x[mask], y[mask], c=intensity[mask], s=0.1)
    plt.show()  
    