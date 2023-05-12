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

import argparse
import os
import openai
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.memory import ConversationBufferMemory
from termcolor import colored

from open_tasks.tools.customized_tools import *

def main():
    
    parser = argparse.ArgumentParser(description='Prepare parameters for running open tasks')
    
    # add arguments to the parser
    parser.add_argument("--searchapi_key", type=str, help="Google search API key", required=True)
    parser.add_argument("--openai_key", type=str, help="OpenAI key", required=True)
    parser.add_argument("--device", type=str, default="cpu")
    
    # parse the arguments
    args = parser.parse_args()
    
    os.environ["SERPAPI_API_KEY"] = args.searchapi_key
    os.environ["OPENAI_API_KEY"] = args.openai_key
    
    openai.api_key = args.openai_key
    
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi","llm-math"],llm=llm)
    
    template="""\
You are a planner who is an expert at coming up with a todo list for a given objective.
For each task, utilize one of the provided tools only when needed. 
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a detailed to-do list to achieve the objective: {objective}

Provided tools:
Text to Animation: useful when you need to input a description text and return a short animation video that matches the text description.
Text to Music: useful when you need to input a description text and return a music that matches the text description.
"""
    
    temp = ""
    for tool in tools:
        temp = temp + tool.name + ": " + tool.description + "\n"
    
    todo_prompt = PromptTemplate.from_template(template+temp)
    
    print (colored("Please specify the task you want to solve: ", 'red', attrs=['bold']))
    objective = input()
    
    prompt = todo_prompt.format(objective=objective)
    
    print(prompt)
    
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    
    content = completion.choices[0].message["content"]
    
    print(content)
    print("\n")
    print (colored("Please decide whether to execute the above plans by typing a number, 0: No, 1: Yes. ", 'red', attrs=['bold']))
    executed = input()
    
    if executed == '1':
        input_list = content.split("\n")[1:-1]
        
        memory = ConversationBufferMemory(memory_key="chat_history")

        agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

        for input_s in input_list:
            output = agent.run(input=(input_s))
            print(output)
        
        print("Finished!")
        
        print("\n")
        print (colored("Please provide a rating for the result, using a scale of 1 to 10.", 'red', attrs=['bold']))
        rating = input()
        
    elif executed == '0':
        print("Finished!")
        
    else:
        print("Invalid Input!")
    
    
if __name__ == "__main__":
    main()
