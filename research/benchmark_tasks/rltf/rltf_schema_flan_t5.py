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

from torch.utils.data import DataLoader
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModel, 
    AutoFeatureExtractor, 
    AutoConfig
)
import os
from benchmark_tasks.generate_model_seq import SeqGen
import torch.optim as optim
from benchmark_tasks.general_dataset import GeneralDataset
from benchmark_tasks.agi_utils import *
from benchmark_tasks.combine_model_seq import SeqCombine

import numpy as np
from IPython.utils import io
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore

import argparse

from undecorated import undecorated
from benchmark_tasks.finetune.utils import construct_optimizer
from types import MethodType



def run_rltf_flan_t5(args):

    """
    load training and test datasets
    """
    data_path = args.data_path
    llm_device = args.llm_device
    eval_device = args.eval_device
    num_seq = args.num_seq
    llm_name = args.llm_name
    
    epochs = args.epochs
    epsilon = args.epsilon
    decay_rate = args.decay_rate

    task_discriptions = txt_loader(data_path + "task_description.txt")
    training_task_idx = [7,20,30,40,50,60]
    # test_task_idx = [2,3,10,15,20,35,45,55,65,70,70,90,106,107]
    test_task_idx = [2]
    training_dataloaders = []
    test_dataloaders = []

    for i in training_task_idx:
        dataset = GeneralDataset(i, data_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        training_dataloaders.append(dataloader)

    for j in test_task_idx:
        dataset = GeneralDataset(j,data_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        test_dataloaders.append(dataloader)

    training_tasks = [task_discriptions[i].strip() for i in training_task_idx]
    test_tasks = [task_discriptions[j].strip() for j in test_task_idx]
    # print(training_tasks)

    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    # Load a pre-trained Vision Transformer model and its feature extractor
    vit_ckpt = "nateraw/vit-base-beans"
    vit = AutoModel.from_pretrained(vit_ckpt)
    vit.eval()
    vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)

    f = transforms.ToPILImage()
    bertscore = load("bertscore")

    seqCombination = SeqCombine(args)

    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    config = AutoConfig.from_pretrained('google/flan-t5-large')
    
    backbone_model = T5ForConditionalGeneration(config=config)
    backbone_model.load_state_dict(torch.load("benchmark_tasks/finetune/10_shot_finetuned.pt", map_location="cpu"))
    backbone_model = backbone_model.to(llm_device)

    seqGen = SeqGen(backbone_model, tokenizer, llm_device)

    generate_with_grad = undecorated(seqGen.model.generate)
    seqGen.model.generate_with_grad = MethodType(generate_with_grad, seqGen.model)
    # optimizer = optim.SGD(seqGen.model.parameters(), lr=0.0001, momentum=0.9)
    optimizer, scheduler = construct_optimizer(args, seqGen.model, 20)

    ## Training
    for e in range(epochs):
        baseline = 0
        rewards = []

        print('num of epoch ' + str(e+1))
        for i, task_description in enumerate(training_tasks):
            task_rewards = []
            # print(task_description)
            optimizer.zero_grad()
            generated_module_seq, log_prob = seqGen.generate_sequence([training_tasks[i]],\
                                                                       module_length=10, \
                                                                       beam_size=30, \
                                                                       num_seq=30,\
                                                                       top_k=5,\
                                                                       top_p=0.5,\
                                                                       temperature=0.9,\
                                                                       constraint=[0,100],\
                                                                       num_beam_groups=1)

            if random.random() >= epsilon:
                action = torch.argmax(torch.stack(log_prob).detach())
            else:
                action = torch.distributions.Categorical(torch.stack(log_prob).detach()).sample()

            # decrease epsilon by the decay rate after each step
            epsilon *= decay_rate

            module_list = generated_module_seq[action][:-1]

            if module_seq_filter(module_list, training_task_idx[i]):

                # print("Module Sequence: " + module_list)
                seqCombination.construct_module_seq(module_list)


                for idx, batch in enumerate(tqdm(training_dataloaders[i])):
                    inputs = list(batch['input'][0])
                    seqCombination.construct_module_seq(module_list)
                    predictions = seqCombination.run_module_seq(inputs)

                    if 0 <= training_task_idx[i] <= 14:
                        outputs = list(batch['output'][0])
                        dist = image_similarity(predictions, outputs, vit, vit_extractor)
                        task_rewards.append(dist/100)
                    elif 15 <= training_task_idx[i] <= 104 or 107 <= task_idx[i]:
                        outputs = list(batch['output'][0])
                        f1 = np.mean(txt_eval(predictions, outputs, bertscore, device=eval_device))
                        task_rewards.append(f1)
                    else:
                        clip_score = score = clip_score(predictions, inputs)
                        task_rewards.append(clip_score.detach()/100)
                ave_task_reward = np.mean(task_rewards)    
                # print("Average reward on current task: " + str(ave_task_reward))
                rewards.append(ave_task_reward)

                seqCombination.close_module_seq()
            else:
                rewards.append(-1)

            avg_reward = np.mean(rewards)
            print("Average reward: " + str(avg_reward))
            loss = -log_prob[action] * (avg_reward - baseline)
            print("Loss: "+ str(loss.item()))
            loss.backward()
            optimizer.step()
            scheduler.step()
            # baseline = avg_reward

    print("Finished training!")    


    ## Testing
    rewards = []
    clips = []
    berts = []
    similairies = []

    for i, task_description in enumerate(test_tasks):
        task_rewards = []
        with torch.no_grad():
            generated_module_seq, log_prob = seqGen.generate_sequence([test_tasks[i]],\
                                                                       module_length=10, \
                                                                       beam_size=30, \
                                                                       num_seq=30,\
                                                                       top_k=5,\
                                                                       top_p=0.5,\
                                                                       temperature=0.9,\
                                                                       constraint=[0,100],\
                                                                       num_beam_groups=1)

        action = torch.argmax(torch.stack(log_prob).detach())


        module_list = generated_module_seq[action][:-1]
        # print(task_description)
        # print("Module Sequence: " + module_list)

        if module_seq_filter(module_list, test_task_idx[i]):
            seqCombination.construct_module_seq(module_list)

            for idx, batch in enumerate(tqdm(test_dataloaders[i])):
                inputs = list(batch['input'][0])
                predictions = seqCombination.run_module_seq(inputs)

                if 0 <= test_task_idx[i] <= 14:
                    outputs = list(batch['output'][0])
                    dist = image_similarity(predictions, outputs, vit, vit_extractor)
                    task_rewards.append(dist/100)
                elif 15 <= test_task_idx[i] <= 104 or 107 <= test_task_idx[i]:
                    outputs = list(batch['output'][0])
                    f1 = np.mean(txt_eval(predictions, outputs, bertscore))

                    task_rewards.append(f1)
                else:
                    score = clip_score(predictions, inputs)
                    task_rewards.append(score.detach()/100)

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

    print("Experimental results: ", np.mean(clips), np.mean(berts), np.mean(similairies), np.mean(rewards))