import os

import json

from .agent_process import (
    AgentProcess,
    # AgentProcessQueue
)

import logging

import time

from threading import Thread

from datetime import datetime

import numpy as np

from ..utils.logger import AgentLogger

from ..utils.chat_template import Query

from abc import ABC, abstractmethod
class CustomizedThread(Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

    def join(self):
        super().join()
        return self.result

class BaseAgent(ABC):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm,
                 agent_process_queue,
                 agent_process_factory,
                 log_mode: str
        ):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.tools = self.config["tools"]
        self.llm = llm
        self.agent_process_queue = agent_process_queue
        self.agent_process_factory = agent_process_factory
        self.tool_list: dict = None

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual" # (mannual, automatic)
        self.rounds = 0

        self.log_mode = log_mode
        self.logger = self.setup_logger()
        self.logger.log(f"Initialized. \n", level="info")

        self.set_status("active")
        self.set_created_time(time.time())

        self.build_system_instruction()

    def run(self):
        '''Execute each step to finish the task.'''
        pass

    # can be customization
    def build_system_instruction(self):
        pass

    def automatic_workflow(self):
        for i in range(self.plan_max_fail_times):
            try:
                response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                    query = Query(
                        messages = self.messages,
                        tools = None
                    )
                )

                if self.rounds == 0:
                    self.set_start_time(start_times[0])

                self.request_waiting_times.extend(waiting_times)
                self.request_turnaround_times.extend(turnaround_times)

                workflow = json.loads(response.response_message)
                self.rounds += 1

                return workflow

            except Exception:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan"
                    }
                )
                continue
        return None

    def manual_workflow(self):
        pass

    def call_tools(self, tool_calls):
        self.logger.log(f"***** It starts to call external tools *****\n", level="info")
        tool_call_responses = None
        for tool_call in tool_calls:
            function_name = tool_call["name"]
            function_to_call = self.tool_list[function_name]
            function_params = tool_call["parameters"]

            try:
                function_response = function_to_call.run(function_params)
                if tool_call_responses is None:
                    tool_call_responses = f"I will call the {function_name} with the params as {function_params} to solve this. The tool response is {function_response}\n"
                else:
                    tool_call_responses += f"I will call the {function_name} with the params as {function_params} to solve this. The tool response is {function_response}\n"

            except Exception:
                continue

        if tool_call_responses:
            self.logger.log(f"At current step, {tool_call_responses}", level="info")

        else:
            self.logger.log("At current step, I fail to call any tools.")

        return tool_call_responses

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, self.agent_name, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    # the default method used for getting response from AIOS
    def get_response(self,
            query,
            temperature=0.0
        ):
        thread = CustomizedThread(target=self.query_loop, args=(query, ))
        thread.start()
        return thread.join()

    def query_loop(self, query):
        agent_process = self.create_agent_request(query)

        completed_response, start_times, end_times, waiting_times, turnaround_times = "", [], [], [], []

        while agent_process.get_status() != "done":
            thread = Thread(target=self.listen, args=(agent_process, ))
            current_time = time.time()
            # reinitialize agent status
            agent_process.set_created_time(current_time)
            agent_process.set_response(None)
            self.agent_process_queue.put(agent_process)

            thread.start()
            thread.join()

            completed_response = agent_process.get_response()
            if agent_process.get_status() != "done":
                self.logger.log(
                    f"Suspended due to the reach of time limit ({agent_process.get_time_limit()}s). Current result is: {completed_response.response_message}\n",
                    level="suspending"
                )
            start_time = agent_process.get_start_time()
            end_time = agent_process.get_end_time()
            waiting_time = start_time - agent_process.get_created_time()
            turnaround_time = end_time - agent_process.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)
            # Re-start the thread if not done

        self.agent_process_factory.deactivate_agent_process(agent_process.get_pid())

        return completed_response, start_times, end_times, waiting_times, turnaround_times

    def create_agent_request(self, query):
        agent_process = self.agent_process_factory.activate_agent_process(
            agent_name = self.agent_name,
            query = query
        )
        agent_process.set_created_time(time.time())
        # print("Already put into the queue")
        return agent_process

    def listen(self, agent_process):
        """Response Listener for agent

        Args:
            agent_process (AgentProcess): Listened AgentProcess

        Returns:
            str: LLM response of Agent Process
        """
        while agent_process.get_response() is None:
            time.sleep(0.2)

        return agent_process.get_response()

    def set_aid(self, aid):
        self.aid = aid

    def get_aid(self):
        return self.aid

    def get_agent_name(self):
        return self.agent_name

    def set_status(self, status):

        """
        Status type: Waiting, Running, Done, Inactive
        """
        self.status = status

    def get_status(self):
        return self.status

    def set_created_time(self, time):
        self.created_time = time

    def get_created_time(self):
        return self.created_time

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time
