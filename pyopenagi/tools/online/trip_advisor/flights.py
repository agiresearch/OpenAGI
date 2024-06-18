from ...base import BaseRapidAPITool

from typing import Any, Dict, List, Optional

from pyopenagi.utils.utils import get_from_env

import requests

import os

import json

class AirportSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/flights/searchAirport"
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
                "The keys in params do not match the excepted keys in params for tripadvisor search airport api. "
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
                'name': result['name'],
                'airportCode': result['airportCode'],
                'coords': result['coords']
            }
            simplified_results.append(simplified_result)

        return json.dumps(simplified_results)




class FlightSearch(BaseRapidAPITool):
    def __init__(self):
        super().__init__()
        self.url = "https://tripadvisor16.p.rapidapi.com/api/v1/flights/searchFlights"
        self.host_name = "tripadvisor16.p.rapidapi.com"
        self.api_key = get_from_env("RAPID_API_KEY")

    def run(self, params: dict):
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host_name
        }

        try:
            self.query_string = {
                "sourceAirportCode": params["sourceAirportCode"],
                "date": params["date"],
                "destinationAirportCode": params["destinationAirportCode"],
                "itineraryType": params["itineraryType"],
                "sortOrder": params["sortOrder"],
                "classOfService": params["classOfService"],
                "returnDate": params["returnDate"]
            }
        except ValueError:
            raise KeyError(
                "The keys in params do not match the excepted keys in params for tripadvisor search flight api. "
                "Please make sure it contains following required keys: "
                "sourceAirportCode",
                "destinationAirportCode",
                "itineraryType",
                "sortOrder",
                "classOfService",
                "returnDate",
                "date"
            )

        response = requests.get(self.url, headers=headers, params=self.query_string).json()
        return self.parse_result(response)

    def parse_result(self, response) -> str:
        # Accessing the 'flights' data from within the 'data' key
        if 'data' in response and 'flights' in response['data']:
            flights_data = response['data']['flights']
            simplified_results = []
            flight_count = 0
            for flight in flights_data:
                if flight_count >= 2:
                    break
                for segment in flight['segments']:
                    for leg in segment['legs']:
                        simplified_result = {
                            'originStationCode': leg['originStationCode'],
                            'destinationStationCode': leg['destinationStationCode'],
                            'departureDateTime': leg['departureDateTime'],
                            'arrivalDateTime': leg['arrivalDateTime'],
                            'classOfService': leg['classOfService'],
                            'marketingCarrierCode': leg['marketingCarrierCode'],
                            'operatingCarrierCode': leg['operatingCarrierCode'],
                            'flightNumber': leg['flightNumber'],
                            'numStops': leg['numStops'],
                            'distanceInKM': leg['distanceInKM'],
                            'isInternational': leg['isInternational']
                        }
                        simplified_results.append(simplified_result)
                flight_count += 1
            return json.dumps(simplified_results)
        else:
            return json.dumps([])
