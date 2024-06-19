from ...base import BaseRapidAPITool

from typing import Any, Dict, List, Optional

# from pydantic import root_validator

from pyopenagi.utils.utils import get_from_env

import requests

class ImdbTopMovieAPI(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://imdb-top-100-movies.p.rapidapi.com/"
        self.host_name = "imdb-top-100-movies.p.rapidapi.com"

        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params):
        start = int(params["start"]) if "start" in params else 1
        end = int(params["end"])
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }
        response = requests.get(self.url, headers=headers).json()
        result = self.parse_result(response, start, end)
        return result


    def parse_result(self, response, start, end) -> str:
        result = []
        # print(response)
        for i in range(start, end):
            item = response[i]
            result.append(f'{item["title"]}, {item["genre"]}, {item["rating"]}, published in {item["year"]}')

        return f"Top {start}-{end} series ranked by IMDB are: " + ";".join(result)
