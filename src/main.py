import dowload as down
import os

folder_m_name = "model/"
model_name = "model.xyz"


if (__name__ == '__main__'):

    
    if(not os.path.exists(folder_m_name)):
        os.mkdir(folder_m_name)    

    if (down.dowload_file("https://disk.yandex.ru/d/N_UY4lDG6TIRTw",folder_m_name + model_name ) ):
        ... 

    # down.download_yandex()