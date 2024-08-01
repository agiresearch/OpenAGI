# stub implementations of tools that we can use (e.g. rapid api)

from openagi.src.utils.utils import get_from_env
import json

class BaseTool:
    """Base class for calling tool
    """
    def __init__(self):
        pass

    def run(self):
        pass
class BaseRapidAPITool(BaseTool):
    """Base class for calling tool from RapidAPI hub: https://rapidapi.com/hub

    Args:
        BaseTool (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.url: str = None
        self.host_name: str = None
        self.api_key: str = None


    def run(self, prompt):
        pass

    def parse_result(self, response: json):
        pass

class BaseHuggingfaceTool(BaseTool):
    def __init__(self):
        pass

    def run(self):
        pass
