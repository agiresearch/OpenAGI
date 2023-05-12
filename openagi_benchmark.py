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

from benchmark_tasks.zero_shot.zero_shot_schema_gpt import run_zero_gpt
from benchmark_tasks.few_shot.few_shot_schema_gpt import run_few_gpt
from benchmark_tasks.few_shot.few_shot_schema_t5 import run_few_flan_t5
from benchmark_tasks.few_shot.few_shot_schema_llama import run_few_llama
from benchmark_tasks.finetune.finetune_schema_flan_t5 import run_finetune_flan_t5
from benchmark_tasks.rltf.rltf_schema_flan_t5 import run_rltf_flan_t5

import argparse
import os 

def main():

    # create the parser object
    parser = argparse.ArgumentParser(description='Process parameters for running benchmark tasks')

    # add arguments to the parser
    parser.add_argument("--openai_key", type=str, help="OpenAI API key")
    parser.add_argument("--huggingface_cache", type=str, help="Cache directory for storing huggingface models and datasets", default="./cache")
    parser.add_argument("--data_path", type=str, help="Data path", required=True)
    parser.add_argument("--device_list", type=str, help="A list of devices for running model sequence", nargs='+')
    parser.add_argument("--eval_device", type=str, help="Device for evaluation")
    parser.add_argument("--llm_device", type=str, help="Device for LLM")
    parser.add_argument("--llm_name", type=str, default="gpt")
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=5)
    parser.add_argument('--task', type=str, help='The object of the task',default="zero_shot")
    
    # arguments only specific for rltf
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_seq", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    
    # parse the arguments
    args = parser.parse_args()
    device_list = []
    
    gpu_ids = args.device_list
    for id in gpu_ids:
        device_list.append("cuda:{}".format(id))
    device_list.append("cpu")
    
    args.device_list = device_list
    
    if args.task == "zero_shot":
        if args.llm_name == "gpt":
            run_zero_gpt(args)
        elif args.llm_name == 'llama':
            pass
        elif args.llm_name == 'flan_t5':
            pass
        else:
            print("Invalid LLM Name")
    
    elif args.task == "few_shot":
        if args.llm_name == "gpt":
            run_few_gpt(args)
        elif args.llm_name == 'llama':
            run_few_llama(args)
        elif args.llm_name == 'flan_t5':
            run_few_flan_t5(args)
        else:
            print("Invalid LLM Name")
            
    elif args.task == "finetune":
        if args.llm_name == 'flan_t5':
            run_finetune_flan_t5(args)
        elif args.llm_name == 'llama':
            pass
        else:
            print("Invalid LLM Name")
    
    elif args.task == "rltf":
        if args.llm_name == 'flan_t5':
            run_rltf_flan_t5(args)
        else:
            print("Invalid LLM Name")
    else:
        print("Invalid Task Name.")
    
    
if __name__ == "__main__":
    main()