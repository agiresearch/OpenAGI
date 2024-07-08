import argparse
import json
import subprocess
import requests
import gzip
import base64
import sys
import os

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["download", "upload"])
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    agent = args.agent

    client = Interactor()
    if mode == "download":
        client.download_agent(agent)
    else:
        assert mode == "upload"
        client.upload_agent(agent)
