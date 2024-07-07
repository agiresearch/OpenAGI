from datetime import datetime
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock, Event
from pympler import asizeof
import os
import json
import subprocess
import requests
import gzip
import base64
import sys
import importlib

class AgentFactory:
    def __init__(self, llm, agent_process_queue, agent_process_factory, agent_log_mode):
        self.max_aid = 256
        self.llm = llm
        self.aid_pool = [i for i in range(self.max_aid)]
        heapq.heapify(self.aid_pool)
        self.agent_process_queue = agent_process_queue
        self.agent_process_factory = agent_process_factory

        self.current_agents = {}

        self.current_agents_lock = Lock()

        self.terminate_signal = Event()

        self.agent_log_mode = agent_log_mode

    def snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    def load_agent_instance(self, script_dir, agent_name, task_input):
        author, name = agent_name.split("/")
        file_path = os.path.join(script_dir, author, name, "agent.py")
        module_name = ".".join(["pyopenagi", "agents", author, name, "agent"])
        class_name = self.snake_to_camel(name)

        print(file_path)
        print(module_name)
        print(class_name)

        agent_module = importlib.import_module(module_name)

        agent_class = getattr(agent_module, class_name)
        return agent_class

    def activate_agent(self, agent_name, task_input):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        if not os.path.exists(os.path.join(script_dir, agent_name)):
            interactor = Interactor()
            interactor.download_agent(agent_name)

        agent_class = self.load_agent_instance(
            script_dir,
            agent_name,
            task_input
        )

        agent = agent_class(
            agent_name = agent_name,
            task_input = task_input,
            llm = self.llm,
            agent_process_queue = self.agent_process_queue,
            agent_process_factory = self.agent_process_factory,
            log_mode = self.agent_log_mode
        )

        aid = heapq.heappop(self.aid_pool)

        agent.set_aid(aid)

        if not self.terminate_signal.is_set():
            with self.current_agents_lock:
                self.current_agents[aid] = agent

        return agent

    def run_agent(self, agent_name, task_input):
        agent = self.activate_agent(
            agent_name=agent_name,
            task_input=task_input
        )
        # print(task_input)
        output = agent.run()
        self.deactivate_agent(agent.get_aid())
        return output

    def print_agent(self):
        headers = ["Agent ID", "Agent Name", "Created Time", "Status", "Memory Usage"]
        data = []
        for id, agent in self.current_agents.items():
            agent_name = agent.agent_name
            created_time = agent.created_time
            status = agent.status
            memory_usage = f"{asizeof.asizeof(agent)} bytes"
            data.append(
                [id, agent_name, created_time, status, memory_usage]
            )
        self.print(headers=headers, data=data)


    def print(self, headers, data):
        # align output
        column_widths = [
            max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))
        ]
        print("+" + "-" * (sum(column_widths) + len(headers) * 3 - 3 ) + "+")
        print(self.format_row(headers, column_widths))
        print("=" * (sum(column_widths) + len(headers) * 3 - 1))
        for i, row in enumerate(data):
            print(self.format_row(row, column_widths))
            if i < len(data):
                print("-" * (sum(column_widths) + len(headers) * 3 - 1))
        print("+" + "-" * (sum(column_widths) + len(headers) * 3 - 3 ) + "+")


    def format_row(self, row, widths, align="<"):
        row_str = " | ".join(f"{str(item):{align}{widths[i]}}" for i, item in enumerate(row))
        return row_str

    def deactivate_agent(self, aid):
        self.current_agents.pop(aid)
        heapq.heappush(self.aid_pool, aid)

class Interactor:
    def __init__(self, base_folder=''):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        self.base_folder = os.path.join(script_dir, base_folder)

    def upload_agent(self, agent):
        agent_dir = os.path.join(self.base_folder, agent)

        author, name = agent.split("/")

        config_file = os.path.join(agent_dir, "config.json")
        with open(config_file, 'r') as f:
            config_data: dict[str, dict] = json.load(f)

        meta_data = config_data.get('meta')

        headers = { "Content-Type": "application/json" }

        # compress python code
        encoded_code = self.compress(
            self.minify_python_code(agent_dir)
        )

        # compress meta_requirements.txt
        encoded_reqs =  self.compress(
            self.minify_reqs(agent_dir)
        )

        # compress config.json
        config_str = json.dumps(config_data)
        encoded_config = self.compress(config_str)

        upload_content = {
            'author': author,
            'name': name,
            'version': meta_data.get('version'),
            'license': meta_data.get('license'),
            'config': encoded_config,
            'code': encoded_code,
            'dependencies': encoded_reqs
        }

        url = 'https://openagi-beta.vercel.app/api/upload'

        response = requests.post(
            url,
            data=json.dumps(upload_content),
            headers=headers
        )
        if response.content:
            print("Uploaded successfully.")

    def minify_python_code(self, agent_dir):
        code_path = os.path.join(agent_dir, "agent.py")
        with open(code_path, 'r') as file:
            lines: list[str] = file.readlines()
        minified_lines = []
        for line in lines:
            stripped_line = line.rstrip()
            if stripped_line and not stripped_line.lstrip().startswith("#"):
                minified_lines.append(stripped_line)
        minified_code = "\n".join(minified_lines)
        return minified_code

    def minify_reqs(self, agent_dir):
        req_path = os.path.join(agent_dir, "meta_requirements.txt")
        with open(req_path, 'r') as file:
            self.reqs: str = file.read()
        cleaned = [line.strip() for line in self.reqs.split(
            '\n') if line.strip() and not line.startswith('#')]
        minified_reqs = ';'.join(cleaned)
        return minified_reqs

    def minify_config(self, config_data):
        minified_config = self.compress(config_data)
        return minified_config

    def compress(self, minified_data):
        compressed_data = gzip.compress(minified_data.encode('utf-8'))
        encoded_data = base64.b64encode(compressed_data)
        encoded_data = encoded_data.decode('utf-8')
        return encoded_data

    # download agent
    def download_agent(self, agent):
        assert "/" in agent, 'agent_name should in the format of "author/agent_name"'
        author, name = agent.split("/")
        # print(author, name)
        query = f'https://openagi-beta.vercel.app/api/download?author={author}&name={name}'
        response = requests.get(query)
        response: dict = response.json()

        if response:
            print("Successfully downloaded")

        agent_folder = os.path.join(self.base_folder, agent)

        if not os.path.exists(agent_folder):
            os.makedirs(agent_folder)

        encoded_config = response.get('config')
        encoded_code = response.get("code")
        encoded_reqs = response.get('dependencies')

        self.download_config(
            self.decompress(encoded_config),
            agent
        )
        self.download_code(
            self.decompress(encoded_code),
            agent
        )
        self.download_reqs(
            self.decompress(encoded_reqs),
            agent
        )

    def decompress(self, encoded_data):
        compressed_data = base64.b64decode(encoded_data)
        decompressed_data = gzip.decompress(compressed_data)
        decompressed_data = decompressed_data.decode("utf-8")
        decompressed_data.replace(";", "\n")
        return decompressed_data

    def download_config(self, config_data, agent) :
        config_path = os.path.join(self.base_folder, agent, "config.json")
        config_data = json.loads(config_data)
        with open(config_path, "w") as w:
            json.dump(config_data, w, indent=4)

    def download_reqs(self, reqs_data, agent):
        reqs_path = os.path.join(self.base_folder, agent, "meta_requirements.txt")

        with open(reqs_path, 'w') as file:
            file.write(reqs_data)

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            reqs_path
        ])

    def download_code(self, code_data, agent):
        code_path = os.path.join(self.base_folder, agent, "agent.py")

        with open(code_path, 'w', newline='') as file:
            file.write(code_data)
