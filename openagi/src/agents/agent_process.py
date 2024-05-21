import os

import time

from queue import Queue

from datetime import datetime

import heapq

from concurrent.futures import ThreadPoolExecutor, as_completed

from threading import Thread, Lock, Event

import threading

from pympler import asizeof

from ..utils.message import Message

import multiprocessing

class LLMRequest:
    def __init__(self, 
            agent_name, agent_id, step,
            message
        ) -> None:
        self.agent_name = agent_name
        self.message = message
        self.agent_id = agent_id
        self.step = step
        self.status = None
        self.response = None
        self.time_limit = None
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
    
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

    def set_priority(self, priority):
        self.priority = priority

    def get_priority(self):
        return self.priority

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_response(self):
        return self.response

    def set_response(self, response):
        self.response = response

    def get_time_limit(self):
        return self.time_limit

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit


class AgentProcess(multiprocessing.Process):
    def __init__(self, agent_name, agent_id, step, message, agent_process_queue, llm_request_responses):
        super().__init__()
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.message = message
        self.step = step
        self.agent_process_queue = agent_process_queue
        self.llm_request_responses = llm_request_responses
        self.response = None
    
    def set_response(self, response):
        self.response = response

    def get_response(self):
        return self.response

    def run(self):
        task_key = (self.agent_id, self.step)

        self.agent_process_queue.put(LLMRequest(self.agent_name, self.agent_id, self.step, self.message))
        
        while True:
            if task_key in self.llm_request_responses:
                status = self.llm_request_responses.get(task_key)["status"]
                if status != "done":
                    self.agent_process_queue.put(LLMRequest(self.agent_name, self.agent_id, self.step, self.message))
                else:
                    break
