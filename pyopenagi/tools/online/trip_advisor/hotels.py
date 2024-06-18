from ...base import BaseRapidAPITool

from typing import Any, Dict, List, Optional

from pyopenagi.utils.utils import get_from_env

import requests

import os

import json

class HotelLocationSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchLocation"
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
                "The keys in params do not match the excepted keys in params for tripadvisor search hotel location api. "
                "Please make sure it contains the key: 'query'"
            )

        # print(self.query_string)

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return json.dumps(response)

    def parse_result(self, response) -> str:
        raise NotImplementedError

class HotelSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchHotels"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }

        try:
            self.query_string = {
                "geoId": params["geoId"],
                "checkIn": params["checkIn"],
                "checkOut": params["checkOut"],
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor search hotel api. "
                "Please make sure it contains following required keys: "
                "geoId",
                "checkIn",
                "checkOut",
            )

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)


    def parse_result(self, response) -> str:
        if 'data' in response and 'data' in response['data']:
            hotels_data = response['data']['data'][:2]
            relevant_info = []
            for hotel in hotels_data:
                relevant_info.append({
                    'id': hotel['id'],
                    'title': hotel['title'],
                    'secondaryInfo': hotel['secondaryInfo'],
                    'bubbleRating': hotel['bubbleRating'],
                    'priceForDisplay': hotel['priceForDisplay'],
                    'priceDetails': hotel['priceDetails'],
                    'priceSummary': hotel['priceSummary']
                })
            return json.dumps(relevant_info)
        else:
            return json.dumps([])

class GetHotelDetails(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/getHotelDetails"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }

        try:
            self.query_string = {
                "id": params["id"],
                "checkIn": params["checkIn"],
                "checkOut": params["checkOut"],
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor get hotel details api. "
                "Please make sure it contains following required keys: "
                "id",
                "checkIn",
                "checkOut",
            )

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)

    def parse_result(self, response) -> str:
        if 'data' in response:
            hotel_data = response['data']
            relevant_info = {
                'name': hotel_data.get('title', ''),
                'rating': hotel_data.get('rating', ''),
                'address': hotel_data.get('location', {}).get('address', ''),
                'amenities': [amenity['title'] for amenity in hotel_data.get('about', {}).get('content', []) if amenity['title'] == 'Amenities'],
                'description': hotel_data.get('about', {}).get('content', [{}])[0].get('content', ''),
                'restaurantsNearby': [{
                    'title': hotel_data.get('restaurantsNearby', {}).get('content', [{}])[0].get('title', ''),
                    'rating': hotel_data.get('restaurantsNearby', {}).get('content', [{}])[0].get('bubbleRating', {}).get('rating', ''),
                    'primaryInfo': hotel_data.get('restaurantsNearby', {}).get('content', [{}])[0].get('primaryInfo', ''),
                    'distance': hotel_data.get('restaurantsNearby', {}).get('content', [{}])[0].get('distance', ''),
                }],
                'attractionsNearby': [{
                    'title': hotel_data.get('attractionsNearby', {}).get('content', [{}])[0].get('title', ''),
                    'rating': hotel_data.get('attractionsNearby', {}).get('content', [{}])[0].get('bubbleRating', {}).get('rating', ''),
                    'primaryInfo': hotel_data.get('attractionsNearby', {}).get('content', [{}])[0].get('primaryInfo', ''),
                    'distance': hotel_data.get('attractionsNearby', {}).get('content', [{}])[0].get('distance', ''),
                }]
            }
            return json.dumps(relevant_info)
        else:
            return json.dumps({})
