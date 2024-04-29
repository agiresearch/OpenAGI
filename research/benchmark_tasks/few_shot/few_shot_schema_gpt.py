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

from sentence_transformers import SentenceTransformer, util
import os
# os.chdir('../')
from benchmark_tasks.general_dataset import GeneralDataset
from torch.utils.data import DataLoader
import torch
from benchmark_tasks.agi_utils import *
import openai
import numpy as np
from IPython.utils import io
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.multimodal import CLIPScore
from benchmark_tasks.combine_model_seq import SeqCombine

def run_few_gpt(args):
    """
    assign openagi data path 
    """
    data_path = args.data_path
    device_list = args.device_list
    eval_device = args.eval_device
    batch_size = args.batch_size
    
    task_discriptions = txt_loader(data_path+"task_description.txt")
    # task_idx = [0,21,61,105,110,120,10,35,62,107,115]
    # test_task_idx = [2,3,10,15,20,35,45,55,65,70,70,90,106,107]
    test_task_idx = [2]
    test_dataloaders = []
    for i in test_task_idx:
        dataset = GeneralDataset(i, data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        test_dataloaders.append(dataloader)

    test_tasks = [task_discriptions[i].strip() for i in test_task_idx]


    # Training
    train_solution = []
    with open(data_path+'train_model_sequence.txt') as f:
        lines = f.readlines()
        for line in lines[:50]:
            train_solution.append(line)
    f.close()

    train_tasks = []
    with open(data_path+'train_task_description.txt') as f:
        lines = f.readlines()
        for line in lines[:50]:
            train_tasks.append(line)
    f.close()

    context = ""
    for i in range(len(train_tasks)):
        steps = ""
        for index,j in enumerate(train_solution[i].split(',')):
            steps += "Step " + str(index+1) + ":" + j.strip("\n") + ", \n"
        cur = "Problem: " + train_tasks[i] +  "Solution:\n" + steps
        context += cur

    # print(context + "Problem: " + test_tasks[0]+"\nSoltuion: ")

    # device_list = ["cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:7","cpu"]
    # device_list = ["cuda:3","cuda:4","cpu"]
    seqCombination = SeqCombine(args)


    # Load a pre-trained Vision Transformer model and its feature extractor
    vit_ckpt = "nateraw/vit-base-beans"
    vit = AutoModel.from_pretrained(vit_ckpt)
    vit.eval()
    vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)

    f = transforms.ToPILImage()
    bertscore = load("bertscore")
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    """
    assign openai api_key
    """
    openai.api_key = args.openai_key

    # device = "cuda:4"
    rewards = []
    clips = []
    berts = []
    similairies = []

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    # Testing
    for i, task_description in enumerate(tqdm(test_tasks)):

        task_rewards = []
        with torch.no_grad():
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",\
                                                      messages=[{"role": "user",\
                                                                 "content": context +\
                                                                 "Problem: " +\
                                                                 task_description +\
                                                                 "\nSoltuion: "}]
                                                      )
            gpt_output = completion.choices[0].message['content'].split('\n')
            gpt_steps = []
            for l,j in enumerate(gpt_output):
                if j[0:4] == "Step":
                    gpt_steps.append(gpt_output[l])
            module_list = match_module_seq(gpt_steps, sentence_model)
        # print(module_list)
        # break


        if len(module_list) > 0 and whole_module_seq_filter(module_list, test_task_idx[i]):
            seqCombination.construct_module_seq(module_list)

            for idx, batch in enumerate(test_dataloaders[i]):
                inputs = list(batch['input'][0])
                predictions = seqCombination.run_module_seq(inputs)

                if 0<=test_task_idx[i]<=14:
                    outputs = list(batch['output'][0])
                    dist = image_similarity(predictions, outputs, vit, vit_extractor)
                    task_rewards.append(dist/100)
                elif 15<=test_task_idx[i]<=104 or 107<=task_idx[i]:
                    outputs = list(batch['output'][0])
                    f1 = np.mean(txt_eval(predictions, outputs, bertscore, device=eval_device))

                    task_rewards.append(f1)
                else:
                    clip_score = clip_score(predictions, inputs)
                    task_rewards.append(clip_score.detach()/100)

            ave_task_reward = np.mean(task_rewards)    


            seqCombination.close_module_seq()

        else:
            ave_task_reward = 0

        if 0 <=test_task_idx[i] <=14:
            similairies.append(ave_task_reward)
        elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

        rewards.append(ave_task_reward)     


    print("Finished testing!")

    print("Evaluation Results: ", np.mean(clips), np.mean(berts), np.mean(similairies), np.mean(rewards))