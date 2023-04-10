""" 
This code is used to support data_augmentation.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/10"
__license__ = "GPLv3"
__version__ = "0.0.1"



import random
import datasets
from PIL import Image, ImageFilter
import skimage
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision


def gaussian_blur(im):
    return im.filter(ImageFilter.GaussianBlur(1))

def gaussian_noise(im):
    gimg = skimage.util.random_noise(np.array(im)/255, mode="speckle", mean=0,var=0.005)
    return Image.fromarray(np.uint8(gimg*255))

def gray_scale(im):
    return im.convert('L').convert('RGB')

def img_augmentation(im,img_aug,input_path):
    if 'grayscale' in img_aug:
        im_1 = gray_scale(im)
    else:
        im_1 = im
    if 'blurry' in img_aug:
        im_2 = gaussian_blur(im_1)
    else:
        im_2 = im_1
    if 'noisy' in img_aug:
        im_3 = gaussian_noise(im_2)
    else:
        im_3 = im_2

    if 'low-resolutioned' in img_aug:
        im_3.convert('RGB')
        im_3.save(input_path+".jpg","JPEG",quality=30)
    else:
        im_3.convert('RGB')
        im_3.save(input_path+".jpg")
    return


def translate2German(sentence,translator):
    translated_res = translator(sentence, max_length=3000)
    return [t['translation_text'] for t in translated_res]
    
def add_mask(sentences):
    mask_sentences = []
    for text in sentences:
        temp = text.split(" ")
        temp[random.randint(0,len(temp)-1)] = '[MASK]'
        mask_sentences.append(' '.join(temp))
    return mask_sentences

def text_augmentation(text,txt_aug,translator):
    if 'clozed' in txt_aug:
        text_1 = add_mask(text)
    else:
        if 'German' in txt_aug:
            text_1 = translate2German(text,translator)  
        else:
            text_1 = text
        
    return text_1