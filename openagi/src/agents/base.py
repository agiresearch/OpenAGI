import os

import json

from .agent_process import (
    AgentProcess,
)

import logging

import time

from threading import Thread

import multiprocessing

from datetime import datetime

import numpy as np

from ..utils.logger import AgentLogger

import threading

class BaseAgent(multiprocessing.Process):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm,
                 agent_process_queue,
                 llm_request_responses,
                 log_mode: str
        ):
        super().__init__()
        self.agent_name = agent_name
        self.config = self.load_config()
        self.prefix = " ".join(self.config["description"])
        self.task_input = task_input
        self.llm = llm
        self.agent_process_queue = agent_process_queue
        self.llm_request_responses = llm_request_responses

        self.log_mode = log_mode
        self.logger = self.setup_logger()
        self.logger.log(f"Initialized. \n", level="info")

        self.set_status("active")
        self.set_created_time(time.time())

    def run(self):
        '''Execute each step to finish the task.'''
        pass

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "agent_config/{}.json".format(self.agent_name))
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    def get_response(self, message, step, temperature=0.0):
        agent_process = AgentProcess(
            agent_name=self.get_agent_name(),
            agent_id=self.get_aid(),
            step=step,
            message = message,
            agent_process_queue=self.agent_process_queue,
            llm_request_responses=self.llm_request_responses
        )
        agent_process.start()
        agent_process.join()

        task_key = (self.get_aid(), step)
        output = self.llm_request_responses.pop(task_key)

        return output

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

    def parse_result(self, prompt):
        pass