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


from evaluate import load
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

def txt_eval(predictions, references, bertscore, device="cuda"):
    score = bertscore.compute(
                    predictions=predictions,
                    references=references,
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=device)["f1"]
    
    return score


def txt_loader(path):
    text = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            text.append(line)
    f.close()
    return text


def image_similarity(im1, im2, model, extractor):
    batch_size = len(im1)
    # Load two images
    img1 = extractor(im1, return_tensors="pt")
    img2 = extractor(im2, return_tensors="pt")

    # Preprocess the images and get their embeddings
    with torch.no_grad():
        emb1 = model(img1.pixel_values)[0].squeeze().numpy()
        emb2 = model(img2.pixel_values)[0].squeeze().numpy()

    # Compute the cosine similarity between the embeddings
    dist = np.mean(np.array([np.linalg.norm(emb1[i] - emb2[i], ord='fro') for i in range(batch_size)]))
    return dist

def module_seq_filter(module_seq, task_id):
    io_dict = { 
                "Colorization":['image','image'],  
                "Image Denoising":['image','image'], 
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],  
                "Image Captioning":['image','text'], 
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],  
                "Text Generation":['text','text'], 
                "Machine Translation":['text','text'],  
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        return True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        return True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        return True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        return True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        return True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        return True
    else:
        return False
    
    
def whole_module_seq_filter(module_seq, task_id):
    io_dict = { 
                "Colorization":['image','image'],  
                "Image Denoising":['image','image'], 
                "Image Deblurring":['image','image'],
                "Image Super Resolution":['image','image'],
                "Image Classification":['image','text'],  
                "Image Captioning":['image','text'], 
                "Object Detection":['image','text'],
                "Text Summarization":['text','text'],  
                "Text Generation":['text','text'], 
                "Machine Translation":['text','text'],  
                "Fill Mask":['text','text'],
                "Sentiment Analysis":['text','text'],
                "Text to Image Generation":['text','image'],
                "Question Answering":['text-text','text'],
                "Visual Question Answering":['image-text','text']
        }
    module_seq_list = module_seq.split(", ")
    condition_1 = None
    for i, m in enumerate(module_seq_list):
        if i < len(module_seq_list)-1 and io_dict[m][1] != io_dict[module_seq_list[i+1]][0]:
            condition_1 = False
            break
        else:
            condition_1 = True
            
        
    condition_2 = None   
    input_type = io_dict[module_seq_list[0]][0]
    output_type = io_dict[module_seq_list[-1]][1]
    if input_type == "image" and output_type == "image" and 0<=task_id<=14:
        condition_2 = True
    elif input_type == "image" and output_type == "text" and 15<=task_id<=104:
        condition_2 = True
    elif input_type == "text" and output_type == "image" and 105<=task_id<=107:
        condition_2 = True
    elif input_type == "text" and output_type == "text" and 108<=task_id<=125:
        condition_2 = True
    elif input_type == "image-text" and output_type == "text" and 126<=task_id<=170:
        condition_2 = True
    elif input_type == "text-text" and output_type == "text" and 171<=task_id<=188:
        condition_2 = True
    else:
        condition_2 = False
        
    return condition_1 and condition_2
    
    
    
def match_module_seq(model_steps, sentence_model):
    module_seq = ""

    for i in range(len(model_steps)):

        sentences1 = [model_steps[i]]*15

        sentences2 = ["Image Classification","Colorization","Object Detection",\
                  "Image Super Resolution","Image Captioning","Image Deblurring",\
                  "Image Denoising","Text to Image Generation","Visual Question Answering",\
                  "Sentiment Analysis","Question Answering","Text Summarization",\
                  "Text Generation","Machine Translation","Fill Mask"]

        #Compute embedding for both lists
        embeddings1 = sentence_model.encode(sentences1, convert_to_tensor=True)#.to(device_)
        embeddings2 = sentence_model.encode(sentences2, convert_to_tensor=True)#.to(device_)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        similarities = torch.stack([cosine_scores[i][i] for i in range(15)])

        module_index = torch.argmax(similarities).item()
        module_seq += sentences2[module_index] + ", "
        # print(similarities[module_index])
        # print(sentences2[module_index])

    #Output the pairs with their score
    # for i in range(len(sentences1)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    module_seq = module_seq.strip()[:-1]
    return module_seq