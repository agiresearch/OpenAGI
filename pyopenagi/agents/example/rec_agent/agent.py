
from ...base import BaseAgent

import time

from ...agent_process import (
    AgentProcess
)

from ....utils.chat_template import Query

from ....tools.online.imdb.top_movie import ImdbTopMovieAPI

from ....tools.online.imdb.top_series import ImdbTopSeriesAPI

import argparse

from concurrent.futures import as_completed

import json

import numpy as np
class RecAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm,
                 agent_process_queue,
                 agent_process_factory,
                 log_mode
        ):
        BaseAgent.__init__(self, agent_name, task_input, llm, agent_process_queue, agent_process_factory, log_mode)

        self.tool_list = {
            "imdb_top_movies": ImdbTopMovieAPI(),
            "imdb_top_series": ImdbTopSeriesAPI()
        }
        # self.workflow = self.config["workflow"]
        # self.tools = self.config["tools"]

    def run(self):
        task_input = "The task you need to solve is: " + self.task_input
        self.logger.log(f"{task_input}\n", level="info")
        request_waiting_times = []
        request_turnaround_times = []

        rounds = 0

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

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            if i == 0:
                self.set_start_time(start_times[0])

            tool_calls = response.tool_calls

            if tool_calls:
                self.logger.log(f"***** It starts to call external tools *****\n", level="info")

                function_responses = ""
                for tool_call in tool_calls:
                    function_name = tool_call["name"]
                    function_to_call = self.tool_list[function_name]
                    function_params = tool_call["parameters"]

                    try:
                        function_response = function_to_call.run(function_params)
                        function_responses += function_response

                        self.messages.append({
                            "role": "assistant",
                            "content": f"I will call the {function_name} with the params as {function_params} to solve this. The tool response is {function_responses}\n"
                        })

                        self.logger.log(f"At current step, it will call the {function_name} with the params as {function_params}. The tool response is {function_responses}\n", level="info")

                    except Exception:
                        continue

                if response_message is None:
                    response_message = function_responses
                    if i == len(self.workflow) - 1:
                        final_result = response_message

            else:
                self.messages.append({
                    "role": "assistant",
                    "content": response_message
                })

                if i == len(self.workflow) - 1:
                    final_result = response_message

                self.logger.log(f"{response_message}\n", level="info")

            rounds += 1

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

    def parse_result(self, prompt):
        return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NarrativeAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    args = parser.parse_args()
    agent = RecAgent(args.agent_name, args.task_input)
    agent.run()
