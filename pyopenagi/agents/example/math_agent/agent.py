
from ...base import BaseAgent

import os

import sys

from ...agent_process import (
    AgentProcess
)

import argparse

from concurrent.futures import as_completed

import numpy as np

from ....tools.online.currency_converter import CurrencyConverterAPI

from ....tools.online.wolfram_alpha import WolframAlpha

from ....utils.chat_template import Query

import time

import json

import re
class MathAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm, agent_process_queue,
                 agent_process_factory,
                 log_mode: str
        ):
        BaseAgent.__init__(self, agent_name, task_input, llm, agent_process_queue, agent_process_factory, log_mode)
        self.tool_list = {
            "wolfram_alpha": WolframAlpha(),
            "currency_converter": CurrencyConverterAPI()
        }

        self.plan_max_fail_times = 3
        self.tool_call_max_fail_times = 3

    def build_system_instruction(self):
        prefix = "".join(
            [
                "".join(self.config["description"]),
                f'You are given the available tools from the tool list: {json.dumps(self.tools)} to help you solve problems.'
            ]
        )
        plan_instruction = "".join(
            [
                'Generate a plan of steps you need to take.',
                'The plan must follow the json format as: '
                '[{"message1": "message_value1","tool_use": [tool_name1, tool_name2,...]}'
                '{"message2": "message_value2", "tool_use": [tool_name1, tool_name2,...]}'
                '...]',
                'In each step of the planned workflow, you must select the most related tool to use'
                'An plan example can be:'
                '[{"message": "identify the tool that you need to call to obtain information.", "tool_use": ["imdb_top_movies", "imdb_top_series"]}]'
                '{"message", "based on the information, give recommendations for the user based on the constrains.", "tool_use": None'
            ]
        )
        exection_instruction = "".join(
            [
                'To execute each step, you need to output as the following json format:',
                '{"observation": "What you have observed from the environment",',
                '"thinking": "Your thought of the current situation",',
                '"action": "Your action to take in current step",',
                '"result": "The result of what you have done."}'
            ]
        )
        if self.workflow_mode == "manual":
            self.messages.append(
                {"role": "system", "content": prefix + exection_instruction}
            )
        else:
            assert self.workflow_mode == "automatic"
            self.messages.append(
                {"role": "system", "content": prefix + plan_instruction + exection_instruction}
            )

    def automatic_workflow(self):
        return super().automatic_workflow()

    def manual_workflow(self):
        workflow = [
            {
                "message": "identify the tool to call to do some pre-calculation. ",
                "tool_use": ["wolfram_alpha", "currency_converter"]
            },
            {
                "message":"perform mathematical operations using the pre-calculated result, which could involve addition, subtraction, multiplication, or division with other numeric values to solve the problem.",
                "tool_use": None
            }
        ]
        return workflow

    def run(self):
        task_input = self.task_input

        self.messages.append(
            {"role": "user", "content": task_input}
        )
        self.logger.log(f"{task_input}\n", level="info")

        workflow = None

        # generate plan
        # workflow = self.automatic_workflow() # generate workflow by llm
        workflow = self.manual_workflow() # define workflow manually

        self.messages.append(
            {"role": "assistant", "content": f"The workflow is {json.dumps(workflow)}"}
        )

        self.logger.log(f"Workflow: {workflow}\n", level="info")

        final_result = ""

        for i, step in enumerate(workflow):
            message = step["message"]
            tool_use = step["tool_use"]

            prompt = f"\nAt step {self.rounds + 1}, you need to {message}. Focus on current step and do not be verbose!"
            self.messages.append({
                "role": "user",
                "content": prompt
            })

            used_tools = self.tools if tool_use else None

            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query = Query(
                    messages = self.messages,
                    tools = used_tools
                )
            )
            if self.rounds == 0:
                self.set_start_time(start_times[0])

            # execute action
            response_message = response.response_message

            tool_calls = response.tool_calls

            self.request_waiting_times.extend(waiting_times)
            self.request_turnaround_times.extend(turnaround_times)

            if tool_calls:
                for i in range(self.plan_max_fail_times):
                    tool_call_responses = self.call_tools(tool_calls=tool_calls)
                    if tool_call_responses:
                        if response_message is None:
                            response_message = tool_call_responses
                        break
                    else:
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": "Fail to call tools correctly. I need to redo the selection of tool and parameters"
                            }
                        )
                        continue

            self.messages.append({
                "role": "assistant",
                "content": response_message
            })

            if i == len(workflow) - 1:
                final_result = response_message

            self.logger.log(f"At step {self.rounds + 1}, {response_message}\n", level="info")

            self.rounds += 1

        self.set_status("done")
        self.set_end_time(time=time.time())

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": self.rounds,
            "agent_waiting_time": self.start_time - self.created_time,
            "agent_turnaround_time": self.end_time - self.created_time,
            "request_waiting_times": self.request_waiting_times,
            "request_turnaround_times": self.request_turnaround_times,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MathAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    args = parser.parse_args()
    agent = MathAgent(args.agent_name, args.task_input)

    agent.run()
