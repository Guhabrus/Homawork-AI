import requests
import yadisk

def dowload_file(_url = "", _out_name = "a.out", batch_size = 1024*1024, prefix_url = "https://getfile.dokpub.com/yandex/get/"):
    """Если хотите скачать с облака то заполняйте префикс - по умолчанию он для Яндекс диска \n
    _url - ссылка на файл \n
    _out_name - имя скачанного файла (пиши с расширением) \n
    batch_size - размер скачанного пакета (не знаешь-не трож) \n
    prefix_url -  url запроса для скачанивания (не знаешь спроси у меня ну или гугли, мне пофиг)""" 

    try:
        response = requests.get(url=(prefix_url+_url), stream=True, allow_redirects=True)
        with open(_out_name, "wb") as file:

            for batch in response.iter_content(chunk_size=batch_size):

                if( batch):
                    file.write(batch)
        print("download succes")
        return True
    except:
        print("Пес, чё то пошло не по плану")
        return False


def download_yandex(_token = "", _url = "", _out_file = "a.out" ):
    """Эту лучше не используй, тут какой то токен нужен, мне в падлу объяснять"""
    
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