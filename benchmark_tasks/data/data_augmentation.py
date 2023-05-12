"""
Copyright 2023 Yingqiang Ge

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

__author__ = "Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/10"
__license__ = "Apache 2.0"
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
from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline
from data_utils import *


def main(args):
    dir_path = args.dir_path+"/openagi_data/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    K = args.num_sample
    device = args.device
    cache_path = args.cache_path
    
    # imagenet_dataset = datasets.load_dataset('frgfm/imagenette','full_size')
    imagenet_dataset = datasets.load_dataset('imagenet-1k','full_size',cache_dir=cache_path, num_proc=6)
    imagenet_imgs = imagenet_dataset['validation']['image']
    imagenet_int2str = imagenet_dataset['validation'].features['label']._int2str
    imagenet_label = imagenet_dataset['validation']['label']
    imagenet_label = [imagenet_int2str[i] for i in imagenet_label]

    cnn_dailymail_dataset = datasets.load_dataset('cnn_dailymail','3.0.0', num_proc=1)
    cnn_summarization = cnn_dailymail_dataset['test']['highlights']
    cnn_article = cnn_dailymail_dataset['test']['article']

    sst_dataset = datasets.load_dataset('sst2', num_proc=1)
    sst_article = sst_dataset['validation']['sentence']
    sst_label = [sst_dataset['validation'].features['label']._int2str[i] for i in sst_dataset['validation']['label']]

    textvqa_dataset = datasets.load_dataset('textvqa', num_proc=1)
    textvqa_q = textvqa_dataset['validation']['question']
    textvqa_a = [i[0] for i in textvqa_dataset['validation']['answers']]
    textvqa_imgs = textvqa_dataset['validation']['image']

    squad_dataset = datasets.load_dataset('squad', num_proc=1)
    squad_context = squad_dataset['validation']['context']
    squad_q = squad_dataset['validation']['question']
    squad_a = [a['text'][0] for a in squad_dataset['validation']['answers']]

    coco_detection_dataset = torchvision.datasets.CocoDetection(args.dir_path+'/coco/val2017', args.dir_path+'/coco/annotations/instances_val2017.json')
    cats = coco_detection_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    a = [i[1] for i in coco_detection_dataset]
    coco_detection_objects = [", ".join([id2label[j['category_id']] for j in i]) for i in a]
    coco_detection_imgs = [i[0] for i in coco_detection_dataset]

    coco_caption_dataset = torchvision.datasets.CocoCaptions(args.dir_path+'/coco/val2017', args.dir_path+'/coco/annotations/captions_val2017.json')
    coco_caption_imgs = [i[0] for i in coco_caption_dataset]
    coco_caption_labels = [i[1][0] for i in coco_caption_dataset]

    input_list = ["image","text","image-text","text-text"]
    output_list = ["image","text"]

    image_color = ["grayscale ",""]
    image_blur = ["blurry ",""] 
    image_noise = ["noisy ", ""]
    image_resolution = ["low-resolutioned ",""]

    text_cloze = ["clozed ",""]
    text_language = ["English "]

    text_output_language = [" in German", " in English"]

    image_output = ["return the regular image"]
    text_output = ["return the summarization","translate the text","return the sentiment"]
    image_text_output = ["return the caption","return the class label","return the object names"]
    text_image_output = ["generate a image"]
    qa_output = ["answer the question"]



    i = "image"
    image_input = []
    for ic in image_color:
        temp_0 = ic + i
        for ib in image_blur:
            temp_1 = ib + temp_0
            for ino in image_noise:
                temp_2 = ino + temp_1
                for ir in image_resolution:
                    temp_3 = ir + temp_2
                    image_input.append(temp_3)

    j = "text"
    text_input = []    
    for tl in text_language:
        j_1 = tl + j

        for tc in text_cloze:
            j_2 = tc + j_1
            text_input.append(j_2)


    input_type_list = []
    for inputs in input_list:
        if inputs == "image":
            input_type = image_input[0:-1]
        if inputs == "text":
            input_type = text_input
        if inputs == "image-text":
            input_type = [img + " and " + tex[0:-5] + " query" for img in image_input[0:-1] for tex in text_input]
        if inputs == "text-text":
            input_type = [tex_1[0:-5] + " document and " + tex_2[0:-5] + " query" for tex_1 in text_input for tex_2 in text_input]

        input_type_list.append(input_type)


    output_type_list = [image_output, \
                        [t + tl for tl in text_output_language for t in text_output if not (tl == ' in German' and t == 'translate the text')],\
                        text_image_output, \
                        [t + tl for tl in text_output_language for t in image_text_output], \
                        [t + tl for tl in text_output_language for t in qa_output]
                       ]


    
    translator = pipeline("translation_en_to_de", model="t5-large", device=device)



    task_id = 0
    task_list = []
    
    for inputs in input_list:
        for outputs in output_list:
            if inputs == "image" and outputs == "image":
                
                input_data = imagenet_imgs[0:K]

                input_type = input_type_list[0]
                output_type = output_type_list[0]
                for i in input_type:
                    for j in output_type:
                        task = "Given " + i + ", how to " + j + " step by step?"
                        task_list.append(task)
                        dataset_id = str(task_id)
                        input_path = dir_path+dataset_id+'/inputs/images/'
                        output_path = dir_path+dataset_id+'/outputs/images/'
                        # Check whether the specified path exists or not
                        isExist_input = os.path.exists(input_path)
                        isExist_output = os.path.exists(output_path)
                        if not isExist_input:
                            os.makedirs(input_path)
                        if not isExist_output:
                            os.makedirs(output_path)

                        img_aug = set(i.split(" ")[0:-1])
                        for k,im in enumerate(input_data):
                            img_augmentation(im,img_aug,input_path+str(k))
                            im.save(output_path+str(k)+".jpg")
                           
                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1

                print('#######Finish image-image')

            if inputs == "text" and outputs == "image":
                """
                text-to-image generation;
                """
                input_text = coco_caption_labels[0:K]

                input_type = input_type_list[1]
                output_type = output_type_list[2]
                for i in input_type:
                    for j in output_type:
                        task = "Given " + i + ", how to " + j + " step by step?"
                        task_list.append(task)
                        dataset_id = str(task_id)

                        input_path = dir_path+dataset_id+'/inputs/'
                        output_path = dir_path+dataset_id+'/outputs/'
                        isExist_input = os.path.exists(input_path)
                        isExist_output = os.path.exists(output_path)
                        if not isExist_input:
                            os.makedirs(input_path)
                        if not isExist_output:
                            os.makedirs(output_path)

                        txt_aug = set(i.split(" ")[0:-1])
                        augmented_text = []
                        augmented_text = text_augmentation(input_text,txt_aug,translator)

                        with open(input_path+'captions.txt', 'w') as fp:
                            for item in augmented_text:
                                # write each item on a new line
                                fp.write("%s\n" % item.strip())
                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1

                print('#######Finish text-image')          


            if inputs == "image" and outputs == "text":
                """
                Object Detection;
                Image Captioning
                """
                input_type = input_type_list[0]
                output_type = output_type_list[3]
                for i in input_type:
                    for j in output_type:
                        task = "Given " + i + ", how to " + j + " step by step?"
                        task_list.append(task)
                        dataset_id = str(task_id)
                        input_path = dir_path+dataset_id+'/inputs/images/'
                        output_path = dir_path+dataset_id+'/outputs/'
                        # Check whether the specified path exists or not
                        isExist_input = os.path.exists(input_path)
                        isExist_output = os.path.exists(output_path)
                        if not isExist_input:
                            os.makedirs(input_path)
                        if not isExist_output:
                            os.makedirs(output_path)

                        txt_aug = set(j.split(" "))
                        if "class" in txt_aug:
                            images = imagenet_imgs[0:K]
                            labels = imagenet_label[0:K]
                        if "caption" in txt_aug:
                            images = coco_caption_imgs[0:K]
                            labels = coco_caption_labels[0:K]
                        if "object" in txt_aug:
                            images = coco_detection_imgs[0:K]
                            labels = coco_detection_objects[0:K]

                        img_aug = set(i.split(" ")[0:-1])
                        for k,im in enumerate(images):
                            img_augmentation(im,img_aug,input_path+str(k))


                        augmented_text = text_augmentation(labels,txt_aug,translator)

                        with open(output_path+'labels.txt', 'w') as fp:
                            for item in augmented_text:
                                # write each item on a new line
                                fp.write("%s\n" % item.strip())


                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1

                print('#######Finish image-text')

            if inputs == "text" and outputs == "text":
                """
                Summary;
                Sentiment Analysis;
                Text Generation;
                Machine Translation;
                Cloze Test;
                """

                input_type = input_type_list[1]
                output_type = output_type_list[1]
                for i in input_type:
                    for j in output_type:
                        task = "Given " + i + ", how to " + j + " step by step?"

                        dataset_id = str(task_id)
                        input_path = dir_path+dataset_id+'/inputs/'
                        output_path = dir_path+dataset_id+'/outputs/'
                        # Check whether the specified path exists or not
                        isExist_input = os.path.exists(input_path)
                        isExist_output = os.path.exists(output_path)
                        if not isExist_input:
                            os.makedirs(input_path)
                        if not isExist_output:
                            os.makedirs(output_path)

                        txt_input_aug = set(i.split(" "))
                        txt_output_aug = set(j.split(" "))

                        if "summarization" in txt_output_aug and "clozed" not in txt_input_aug:
                            texts = cnn_article[0:K]
                            labels = cnn_summarization[0:K]


                        elif "translate" in txt_output_aug:
                            texts = coco_caption_labels[0:K]
                            labels = coco_caption_labels[0:K]

                        elif "sentiment" in txt_output_aug:
                            texts = sst_article[0:K]
                            labels = sst_label[0:K]
                        else:
                            continue

                        task_list.append(task)

                        augmented_input_text = text_augmentation(texts, txt_input_aug,translator)

                        with open(input_path+'text.txt', 'w') as fp:
                            for item in augmented_input_text:
                                # write each item on a new line
                                fp.write("%s\n" % item.strip())


                        augmented_output_text = text_augmentation(labels, txt_output_aug,translator)

                        with open(output_path+'labels.txt', 'w') as fp:
                            for item in augmented_output_text:
                                # write each item on a new line
                                fp.write("%s\n" % item.strip())


                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1
                print('#######Finish text-text')           

            if inputs == "image-text" and outputs == "text":
                """
                VQA
                """
                # continue
                input_imgs = textvqa_imgs[0:K]
                input_q = textvqa_q[0:K]
                output_a = textvqa_a[0:K]

                input_type = input_type_list[2]
                for i in input_type:
                    for j in output_type_list[-1]:
                        task = "Given " + i + ", how to " + j + " step by step?"
                        task_list.append(task)
                        dataset_id = str(task_id)

                        input_img_path = dir_path+dataset_id+'/inputs/images/'
                        input_q_path = dir_path+dataset_id+'/inputs/'
                        output_a_path = dir_path+dataset_id+'/outputs/'
                        # Check whether the specified path exists or not
                        isExist_input_img = os.path.exists(input_img_path)
                        isExist_input_q = os.path.exists(input_q_path)
                        isExist_output_a = os.path.exists(output_a_path)
                        if not isExist_input_q:
                            os.makedirs(input_q_path)
                        if not isExist_input_img:
                            os.makedirs(input_img_path)   
                        if not isExist_output_a:
                            os.makedirs(output_a_path)


                        img_input_aug = i.split("and")[0].strip()
                        txt_input_aug = i.split("and")[1].strip()

                        for k,im in enumerate(input_imgs):
                            img_augmentation(im,img_input_aug,input_img_path+str(k))

                        augmented_input_text = text_augmentation(input_q,txt_input_aug,translator)

                        with open(input_q_path+'questions.txt', 'w') as fp:
                            for item in augmented_input_text:
                                fp.write("%s\n" % item.strip())


                        txt_output_aug = set(j.split(" "))

                        augmented_output_text = text_augmentation(output_a,txt_output_aug,translator)

                        with open(output_a_path+'answers.txt', 'w') as fp:
                            for item in augmented_output_text:
                                fp.write("%s\n" % item.strip())

                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1
                print('#######Finish VQA')            

            if inputs == "text-text" and outputs == "text":
                """
                QA;
                """
                input_context = squad_context[0:K]
                input_q = squad_q[0:K]
                output_a = squad_a[0:K]
                # continue
                input_type = input_type_list[3]
                for i in input_type:
                    for j in output_type_list[-1]:
                        task = "Given " + i + ", how to " + j + " step by step?"
                        task_list.append(task)

                        dataset_id = str(task_id)

                        input_c_path = dir_path+dataset_id+'/inputs/'
                        input_q_path = dir_path+dataset_id+'/inputs/'
                        output_a_path = dir_path+dataset_id+'/outputs/'
                        # Check whether the specified path exists or not
                        isExist_input_q = os.path.exists(input_q_path)
                        isExist_output_a = os.path.exists(output_a_path)
                        if not isExist_input_q:
                            os.makedirs(input_q_path) 
                        if not isExist_output_a:
                            os.makedirs(output_a_path)

                        input_c_aug = i.split("and")[0].strip()
                        input_q_aug = i.split("and")[1].strip()


                        augmented_input_q = text_augmentation(input_q,input_q_aug,translator)

                        with open(input_q_path+'questions.txt', 'w') as fp:
                            for item in augmented_input_q:
                                fp.write("%s\n" % item.strip())


                        augmented_input_c = text_augmentation(input_context,input_c_aug,translator)

                        with open(input_c_path+'context.txt', 'w') as fp:
                            for item in augmented_input_c:
                                fp.write("%s\n" % item.strip())

                        output_a_aug = set(j.split(" "))

                        augmented_output_a = text_augmentation(output_a, output_a_aug,translator)

                        with open(output_a_path+'answers.txt', 'w') as fp:
                            for item in augmented_output_a:
                                fp.write("%s\n" % item.strip())

                        print('Finish creating dataset for task '+str(task_id))
                        task_id += 1

                print('#######Finish QA')


    print('Total number of tasks: ' +str(len(task_list)))    
    
    with open(dir_path+"/task_description.txt", "w") as output:
        for i,row in enumerate(task_list):
            output.write(row + '\n')
           
        
        
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--dir_path", type=str, default="/common/users/yg334")
    parser.add_argument("--cache_path", type=str, default="/common/users/yg334/LLAMA/huggingface/cache/")
    parser.add_argument("--num_sample", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
