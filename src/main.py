import os

import dowload as down
import algoritm as alg

folder_m_name = "../model/"
model_name = "model.xyz"


import numpy as np

if (__name__ == '__main__'):

    
    # if(not os.path.exists(folder_m_name)):
    #     os.mkdir(folder_m_name)    

    # if (down.dowload_file("https://disk.yandex.ru/d/N_UY4lDG6TIRTw",folder_m_name + model_name ) ):
    #     ... 

    # alg.data_process(folder_m_name + model_name)

    x = np.arange(10)
    y = np.arange(10, 20)
    

    
    x = np.append(x,y, axis=0)
    x = np.reshape(x, (2,len(y)))
    
    print(x)
    