

import random
#import zipfile
from copy import deepcopy
#from pathlib import Path
#import matplotlib.pyplot as plt
#import numpy as np
import torch
from PIL import Image
from torch import nn
#from torch import optim

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from torchvision import transforms
#from torchvision import datasets

import time
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2
import os
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#МОДЕЛЬ 1.
#small 
#обучена на 20000 на класса  256x256
#точность 96.63 минимальный лосс 0.10854
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),

#модель 2.
#small 
#обучена на 3000 на класса  320х320
#точность 95.06 минимальный лосс 0.15
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),

#модель 3.
#BASE 
#обучена на 20000 на класса  266х266
#точность 97.09 минимальный лосс 0.08456  !!!!!!!!!!!!!!!!!!!!!!!
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(256, len(class_names))) 





#device = 'cpu'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('ВНИМАНИЕ!!! Обрабатываеться будет всё на ',device, '<<<<<<<<<<<<<----------------')

cls = {0:'hard',1 :'control', 2:'sexy'}
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')


from dinov2.models.vision_transformer import vit_small, vit_base #vit_large, vit_giant2
class DinoVisionTransformerClassifier(nn.Module):

    def __init__(self, model_size="small"):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model_size = model_size

        # loading a model with registers
        n_register_tokens = 4

        if model_size == "small":
            model = vit_small(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 384
            self.number_of_heads = 6
            #model.load_state_dict(torch.load('models/dinov2_vits14_reg4_pretrain.pth',map_location=torch.device('cpu')))
            self.transformer = deepcopy(model)
            self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),

        elif model_size == "base":
            model = vit_base(patch_size=14,
                             img_size=526,
                             init_values=1.0,
                             num_register_tokens=n_register_tokens,
                             block_chunks=0,
                            #drop_path_rate=0.05
                            )
            self.embedding_size = 768
            self.number_of_heads = 12
            self.transformer = deepcopy(model)
            self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(256, 3))

            #model.load_state_dict(torch.load('models/dinov2_vitb14_reg4_pretrain.pth'))

        

        


    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
model1 = DinoVisionTransformerClassifier("small")
#model2 = DinoVisionTransformerClassifier("small")
model2 = DinoVisionTransformerClassifier("base")

model1.load_state_dict(torch.load('model1.pth', map_location=torch.device(device)))
#model2.load_state_dict(torch.load('model2.pth', map_location=torch.device(device)))
model2.load_state_dict(torch.load('model3.pth', map_location=torch.device(device)))


model1.eval()
model2.eval()



class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        # Resize the image
        img = v2.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = v2.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)

        return img



inference_preprocessing1 = transforms.Compose([ ResizeAndPad(256, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )
inference_preprocessing2 = transforms.Compose([ ResizeAndPad(320, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )

inference_preprocessing3 = transforms.Compose([ ResizeAndPad(266, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )


def get_predict_from_model(img, model, preprocess):
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    input_tensor = img_tensor.to(device)
    embeddings = model.transformer(input_tensor)
    x = model.transformer.norm(embeddings)
    output_tensor = model.classifier(x)
    probabilities = torch.softmax(output_tensor, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

    c = int(predicted_classes[0])
    
    rez = dict(zip(cls.values(), probabilities.flatten().tolist()))
    return  rez



def look_to_file(img):
    r1 = get_predict_from_model(img, model1, inference_preprocessing1)
    r2 = get_predict_from_model(img, model2, inference_preprocessing3)
    
    #img = Image.open(file) 
    #img_tensor = inference_preprocessing(img)
    #img_tensor = img_tensor.unsqueeze(0)
    #input_tensor = img_tensor.to(device)
    #embeddings = model.transformer(input_tensor)
    #x = model.transformer.norm(embeddings)
    #output_tensor = model.classifier(x)
    #probabilities = torch.softmax(output_tensor, dim=1)
    #predicted_classes = torch.argmax(probabilities, dim=1)

    #c = int(predicted_classes[0])
    
    #rez = dict(zip(cls.values(), probabilities.flatten().tolist()))
    if (r1['sexy'] > 0.85) & (r2['sexy'] > 0.85):
        need_moderation = 0
    else:
        need_moderation = 1

    rez = {}
    rez['need_moderation'] = need_moderation
    rez['net1'] = r1
    rez['net2'] = r2
    
    return rez
