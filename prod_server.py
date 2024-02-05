#from fastapi import FastAPI, File, UploadFile,Form,  Depends
from xml.dom import WrongDocumentErr
from fastapi import FastAPI, File, UploadFile,Depends

from typing import Optional
from PIL import Image
from io import BytesIO
from pydantic import BaseModel#это понадобилось для втрого варианта
#нужно для работы с файлами с айфона

from pillow_heif import register_heif_opener

from datetime import datetime
import os

#import imghdr
#пробуем вставить логирование всех запросов, которые приходят на сервер
import time
from typing import Callable
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.routing import APIRoute

import gc

#class TimedRoute(APIRoute):
#    def get_route_handler(self) -> Callable:
#        original_route_handler = super().get_route_handler()
#        async def custom_route_handler(request: Request) -> Response:
#            before = time.time()
#            response: Response = await original_route_handler(request)

     
#            current_datetime = datetime.now()
#            str_current_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")

#            duration = time.time() - before
#            response.headers["X-Response-Time"] = str(duration)
          
#            #print(str_current_datetime, request.url.scheme, request.method, request.url.path, request.query_params, f'{request.client.host}:{request.url.port}', duration)
            
#            return response
#        return custom_route_handler





register_heif_opener()

import prod #в этом  файле хранятся все процедуры

app = FastAPI()
#router = APIRouter(route_class=TimedRoute) #ЭТО ДЛЯ РОУТЕРА
#ПОЛУЧЕНИЕ ТОЛЬКО КАРТИНКИ. ОТВЕТ json
keys = ['hard', 'control', 'sexy']
wrong = {'need_moderation': 3, 'net1': {'hard': 0.0, 'control': 0.0, 'sexy': 0.0}, 'net2': {'hard': 0.0, 'control': 0.0, 'sexy': 0.0}}

class Base2(BaseModel):
    foto_id: Optional[int] = 1    #Требуется вернуть тут будут ID фото из внутренней базы
    detect: Optional[int] = 1    #необходима детекция объектов
    classify: Optional[int] = 1    #необходима классификация  изображений


@app.post("/ps")
async def analyze_image1(image: UploadFile = File(...)):
#async def analyze_image(image: UploadFile = File(...)):


#async def analyze_image(base: Base2 = Depends(), image: UploadFile = File(...)):
    #print('имя файла',image.filename)
    #image_type = imghdr.what(image.file)
    #if image_type or image.filename.lower().endswith('heic'):
    if True : #тут можно проверить изображение ли это вообще...см.строку выше
        with Image.open(BytesIO(await image.read())) as img:
            #print('ТУТ')
            #print("width", img.width, "height", img.height)
            #print('ТУТ2')
            if img.mode != 'RGB':
                img = img.convert("RGB")
            #try:
            #return 1
            #print('тут3')
            pok = prod.look_to_file(img)#отправляем изображение и то, что требуется вернуть
            #except:
             #   pok = wrong
                #print(pok)
			
			#можно сохранить полученное изображение в arh
            #otvet = [f'{round(pok["net1"][key],2)}_{round(pok["net2"][key],2)} '   for key in keys]
            #otvet = '__'.join(otvet)
            #print(otvet)

            #need_moderation = pok['need_moderation']
            #current_datetime = datetime.now()
            #str_current_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S.%f")
        
            #filename = f"need_mod_{need_moderation}__{otvet}__{str_current_datetime}.jpg"
            
            #filename = os.path.join('arh',filename)
            #img.save(filename)
            #print('Сохранено  ',filename)

			
            gc.collect() #почистимся 
            return pok
    else:
         return None

@app.post("/test")
async def analyze_image(image: UploadFile = File(...)):
    with Image.open(BytesIO(await image.read())) as img:
         current_datetime = datetime.now()
         str_current_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
         filename = f"inp_{str_current_datetime}.jpg"
         filename = os.path.join('arh', filename)
         #img.save(filename)
         return {"width": img.width, "height": img.height}



txt = 'Этот адрес url не предназначен для открытия в браузере. Используйте post запрос в формате requests.post(url, files=file) где - file - это бинарный файл с изображением для распознавания'



@app.get("/ps")
async def not_use():
   
   
    return txt

@app.get("/test")
async def not_use():
   
    return txt

#app.include_router(router)