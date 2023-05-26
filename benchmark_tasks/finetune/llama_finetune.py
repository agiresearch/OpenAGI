""" 
This code mainly used to fine-tune Flan-T5.

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

__author__ = "Wenyue Hua, Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/09"
__license__ = "GPLv3"
__version__ = "0.0.1"


import os
import torch
from torch import optim
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from utils import set_seed, Logger, construct_optimizer
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from generate_trie import generate_sequence, candidates
# from generate_parallel_trie import generate_sequence, candidates
# from generate_tree_structure_t5 import generate_sequence, candidates
from generate_parallel_trie_llama import generate_sequence, candidates
from undecorated import undecorated
from types import MethodType
import random
import json

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)



def load_data(args, tokenizer):
    with open(args.prompt_dir, "r") as f:
        prompts = f.read()
    prompts = prompts.split("\n")[:-1]

    with open(args.answer_dir, "r") as f:
        answers = f.read()
    answers = answers.split("\n")[:-1]

    data = [(prompt, answer) for prompt, answer in zip(prompts, answers)]
    # random.shuffle(data)
    with open("finetune_train.json", "w") as f:
        json.dump(data[: args.num_train], f)

    # tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    class InputDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.prompts = [d[0]+" "+d[1] for d in data]
            self.answers = [d[0]+" "+d[1] for d in data]

        def __len__(self):
            return len(self.answers)

        def __getitem__(self, index):
            prompts = tokenizer(self.prompts[index], return_tensors="pt")
            answers = tokenizer(self.answers[index], return_tensors="pt")

            # return {
            #     "instruction": self.prompts[index],
            #     "input": "",
            #     "output": self.answers[index]
            # }
            return {
                "input_ids": prompts["input_ids"].squeeze(),
                "attention_mask": prompts["attention_mask"].squeeze(),
                "labels": answers["input_ids"].squeeze(),
                   }
            

    train_dataset = InputDataset(data[: args.num_train])
    test_dataset = InputDataset(data[args.num_train :])

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataset, test_dataset


def train(args, logger, max_memory_mapping):
    logger.log("load model")
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map="auto",
    )
    # generate_with_grad = undecorated(model.generate)
    # model.generate_with_grad = MethodType(generate_with_grad, model)

    # tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        cache_dir = args.cache_dir,
    )
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    
    model = prepare_model_for_int8_training(model)
    
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj","v_proj"]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    
    train_loader, test_loader = load_data(args, tokenizer)
    # optimizer, scheduler = construct_optimizer(args, model, 20)
    micro_batch_size = 1
    gradient_accumulation_steps = args.batch_size // micro_batch_size
    val_set_size = 8
    output_dir = "/common/users/yg334/lora-vicuna/"
    group_by_length = False
    # ddp = False
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        
    gradient_accumulation_steps = gradient_accumulation_steps // world_size
        
    use_wandb = False
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_loader,
        eval_dataset=test_loader,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=10000,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    trainer.train(resume_from_checkpoint=False)
    
    model.save_pretrained(output_dir)

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default="toy.log")
    parser.add_argument("--model_name", type=str, default="eachadea/vicuna-7b-1.1")
    parser.add_argument("--cache_dir", type=str, default="/common/users/yg334/LLAMA/huggingface/cache")
    parser.add_argument("--prompt_dir", type=str, default="train_task_description.txt")
    parser.add_argument("--answer_dir", type=str, default="train_model_sequence.txt")

    parser.add_argument("--toy", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_seq", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)

    parser.add_argument("--num_train", type=int, default=15)

    args = parser.parse_args()

    set_seed(args)
    logger = Logger(args.logging_dir)
    

    train(args, logger, max_memory_mapping=None)
    
    
