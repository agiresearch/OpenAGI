
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

import time

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
            # "wolfram_alpha": WolframAlpha(),
            # "currenct_converter": CurrencyConverterAPI()
        }
        self.tool_check_max_fail_times = 10
        self.tool_select_max_fail_times = 10
        self.tool_calling_max_fail_times = 10
        self.tool_info = "".join(self.config["tool_info"])

    def load_flow(self):
        return

    def run(self):
        prompt = ""
        prefix = self.prefix
        task_input = self.task_input
        prompt += prefix
        request_waiting_times = []
        request_turnaround_times = []
        task_input = "The task you need to solve is: " + task_input
        prompt += task_input

        self.logger.log(f"{task_input}\n", level="info")

        rounds = 0
        # predefined steps
        steps = [
            "identify and outline the sub-problems that need to be solved as stepping stones toward the solution. ",
            "solve each sub-problem. ",
            "integrate the solutions to these sub-problems in the previous step to get the final solution. "
        ]
        for i, step in enumerate(steps):
            prompt += f"\nIn step {i+1}, you need to {step}. Output should focus on current step and don't be verbose!"

            self.logger.log(f"Step {i+1}: {step}\n", level="info")
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(prompt)

            if rounds == 0:
                self.set_start_time(start_times[0])

            rounds += 1

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)
            prompt += f"The solution to step {i+1} is: {response}\n"
            self.logger.log(f"The solution to step {i+1}: {response}\n", level="info")

        prompt += f"Given the interaction history: '{prompt}', integrate solutions in all steps to give a final answer, don't be verbose!"

        final_result, start_times, end_times, waiting_times, turnaround_times = self.get_response(prompt)
        request_waiting_times.extend(waiting_times)
        request_turnaround_times.extend(turnaround_times)

        self.set_status("done")
        self.set_end_time(time=time.time())

        self.logger.log(
            f"{task_input} Final result is: {final_result}\n",
            level="info"
        )
        
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
