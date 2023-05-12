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
import transformers

import os
# os.chdir('../')
from benchmark_tasks.general_dataset import GeneralDataset
from torch.utils.data import DataLoader
import torch
from benchmark_tasks.agi_utils import *
import torch

import numpy as np
from IPython.utils import io
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.multimodal import CLIPScore
from benchmark_tasks.combine_model_seq import SeqCombine


def run_few_llama(args):
    data_path = args.data_path
    device_list = args.device_list
    eval_device = args.eval_device
    batch_size = args.batch_size
    
    task_discriptions = txt_loader(data_path+"task_description.txt")
    # task_idx = [0,21,61,105,110,120,10,35,62,107,115]
    # test_task_idx = [2,3,10,15,20,35,45,55,65,70,90,106,107]
    test_task_idx = [2]
    test_dataloaders = []
    for i in test_task_idx:
        dataset = GeneralDataset(i, data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        test_dataloaders.append(dataloader)

    test_tasks = [task_discriptions[i].strip() for i in test_task_idx]
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


    # max_memory_mapping = {0: "18GB", 1: "18GB", 2: "18GB", 3: "18GB", 4: "18GB", 5: "18GB", 6: "18GB", 7: "18GB",}
    max_memory_mapping = {0: "0GB", 1: "10GB", 2: "24GB", 3: "0GB", 4: "0GB", 5: "0GB", 6: "24GB", 7: "0GB",}

    llama_tokenizer = transformers.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    llama = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", \
                                                          # load_in_8bit=True,\
                                                          device_map='auto',\
                                                          max_memory = max_memory_mapping)

    


    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")


    seqCombination = SeqCombine(args)

    # Load a pre-trained Vision Transformer model and its feature extractor
    vision_model_ckpt = "nateraw/vit-base-beans"
    vision_model = AutoModel.from_pretrained(vision_model_ckpt)
    vision_model.eval()
    vision_extractor = AutoFeatureExtractor.from_pretrained(vision_model_ckpt)

    f = transforms.ToPILImage()
    bertscore = load("bertscore")

    rewards = []
    clips = []
    berts = []
    similairies = []

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")


    for i, task_description in enumerate(tqdm(test_tasks)):

        task_rewards = []
        with torch.no_grad():        
            prompt = context + "Problem: " + task_description + "\nSoltuion: "
            llama_inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda:1")
            generated_ids = llama.generate(**llama_inputs, max_length=1000)
            llama_outputs = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            llama_solution = llama_outputs[0].split("Problem: ")[1]
            llama_solution = llama_solution.split("\n")

            llama_steps = []
            for l,j in enumerate(llama_solution):
                if j[0:4] == "Step":
                    llama_steps.append(llama_solution[l])

            module_list = match_module_seq(llama_steps, sentence_model)


        if len(module_list) > 0 and whole_module_seq_filter(module_list, test_task_idx[i]):
            seqCombination.construct_module_seq(module_list)

            for idx, batch in enumerate(test_dataloaders[i]):
                inputs = list(batch['input'][0])
                predictions = seqCombination.run_module_seq(inputs)

                if 0<=test_task_idx[i] <= 14:
                    outputs = list(batch['output'][0])
                    dist = image_similarity(predictions, outputs, vision_model, vision_extractor)
                    task_rewards.append(dist/100)
                elif 15 <= test_task_idx[i] <= 104 or 107 <= task_idx[i]:
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

        if 0 <= test_task_idx[i] <= 14:
            similairies.append(ave_task_reward)
        elif 15 <= test_task_idx[i] <= 104 or 107 <= test_task_idx[i]:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

        rewards.append(ave_task_reward)     


    print("Finished testing!")

    print("Evaluation results: ", np.mean(clips), np.mean(berts), np.mean(similairies), np.mean(rewards))