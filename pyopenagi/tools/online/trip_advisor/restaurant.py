from ...base import BaseRapidAPITool

from typing import Any, Dict, List, Optional

from pyopenagi.utils.utils import get_from_env

import requests

import os

import json

class RestaurantLocationSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchLocation"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }
        try:
            self.query_string = {
                "query": params["query"],
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor search restaurant location api. "
                "Please make sure it contains the key: 'query'"
            )

        # print(self.query_string)

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)

    def parse_result(self, response) -> str:
        limited_results = response['data'][:2]

        simplified_results = []
        for result in limited_results:
            simplified_result = {
                'locationId': result['locationId'],
                'localizedName': result['localizedName'],
                'latitude': result['latitude'],
                'longitude': result['longitude']
            }
            simplified_results.append(simplified_result)

        return json.dumps(simplified_results)


class RestaurantSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchRestaurants"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }

        try:
            self.query_string = {
                "locationId": params["locationId"]
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor search restaurant api. "
                "Please make sure it contains following required keys: "
                "locationID",
            )

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)

    def parse_result(self, response) -> str:
        limited_results = response['data']['data'][:2]

        simplified_results = []
        for result in limited_results:
            simplified_result = {
                'restaurantsId': result['restaurantsId'],
                'name': result['name'],
                'averageRating': result['averageRating'],
                'userReviewCount': result['userReviewCount'],
                'priceTag': result['priceTag'],
                'establishmentTypeAndCuisineTags': result['establishmentTypeAndCuisineTags']
            }
            simplified_results.append(simplified_result)

        return json.dumps(simplified_results)


class GetRestaurantDetails(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/getRestaurantDetails"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }

        try:
            self.query_string = {
                "restaurantsId": params["restaurantsId"],
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor get restaurant details api. "
                "Please make sure it contains the key: 'restaurantsID'"
            )

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)


    def parse_result(self, response) -> str:
        location = response["data"]["location"]

        useful_info = {
            "name": location.get("name"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "num_reviews": location.get("num_reviews"),
            "rating": location.get("rating"),
            "price_level": location.get("price_level"),
            "address": location.get("address"),
            "phone": location.get("phone"),
            "website": location.get("website"),
            "cuisine": [cuisine["name"] for cuisine in location.get("cuisine", [])],
            "hours": location.get("hours", {}).get("week_ranges", [])
            }
        return json.dumps(useful_info)
