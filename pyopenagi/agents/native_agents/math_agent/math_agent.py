
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

    def load_flow(self):
        return

    def run(self):
        task_input = self.task_input
        request_waiting_times = []
        request_turnaround_times = []

        self.logger.log(f"{task_input}\n", level="info")

        rounds = 0

        # predefined steps
        # TODO replace fixed steps with dynamic steps
        # Step 1 "use a currency converter tool to get the currency information. ",
        # Step 2 "perform mathematical operations using the converted currency amount, which could involve addition, subtraction, multiplication, or division with other numeric values to solve the problem."

        for i, step in enumerate(self.workflow):

            prompt = f"\nAt current step, you need to {step}. Output should focus on current step and don't be verbose!"
            self.messages.append({
                "role": "user",
                "content": prompt
            })
            tool_use = self.tools if i == 0 else None
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query = Query(
                    messages = self.messages,
                    tools = tool_use
                )
            )
            response_message = response.response_message
            if i == 0:
                self.set_start_time(start_times[0])

            tool_calls = response.tool_calls

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            if tool_calls:
                self.logger.log(f"***** It starts to call external tools *****\n", level="info")

                function_responses = ""
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.tool_list[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    try:
                        function_response = function_to_call.run(function_args)
                        function_responses += function_response

                        self.messages.append({
                            "role": "user",
                            "content": f"It calls the {function_name} with the params as {function_args} to solve this. The tool response is {function_responses}\n"
                        })

                        self.logger.log(f"For current step, it will call the {function_name} with the params as {function_args}. The tool response is {function_responses}\n", level="info")

                    except Exception:
                        continue

                if response_message is None:
                    response_message = function_responses
                    if i == len(self.workflow) - 1:
                        final_result = response_message

            else:
                self.messages.append({
                    "role": "user",
                    "content": response_message
                })

                if i == len(self.workflow) - 1:
                    final_result = response_message

                self.logger.log(f"{response_message}\n", level="info")

            round += 1

        self.set_status("done")
        self.set_end_time(time=time.time())

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": rounds,
            "agent_waiting_time": self.start_time - self.created_time,
            "agent_turnaround_time": self.end_time - self.created_time,
            "request_waiting_times": request_waiting_times,
            "request_turnaround_times": request_turnaround_times,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MathAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    args = parser.parse_args()
    agent = MathAgent(args.agent_name, args.task_input)

    agent.run()
    # thread_pool.submit(agent.run)
    # agent.run()
