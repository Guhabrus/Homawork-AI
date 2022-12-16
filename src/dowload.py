import requests
import yadisk

def dowload_file(_url = "", _out_name = "a.out", batch_size = 1024*1024):
    try:
        response = requests.get(url=_url, stream=True, allow_redirects=True)
        with open(_out_name, "wb") as file:

            for batch in response.iter_content(chunk_size=batch_size):

                if( batch):
                    file.write(batch)
        return True
    except:
        print("Пес, чё то пошло не по плану")
        return False


def download_yandex(_token = "", _url = "", _out_file = "a.out" ):
    
    
    disk = yadisk.YaDisk(token=_token)
    if( not disk.check_token() ):
        print(" Твоё токен гавно!!!")
        return False

    try:
        disk.download(path_or_file= _out_file)
    except:
        print("Сорян, но чет не получается скачать, хз почему")
        return False
    
    return True