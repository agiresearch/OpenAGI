import json
import os
import sys
import argparse
import time
import numpy as np
from concurrent.futures import as_completed
from ....utils.message import Message

from ...base import BaseAgent
from ....tools.online.trip_advisor.hotels import HotelLocationSearch, HotelSearch, GetHotelDetails
from ....tools.online.trip_advisor.flights import AirportSearch, FlightSearch
from ....tools.online.trip_advisor.restaurant import RestaurantLocationSearch, RestaurantSearch, GetRestaurantDetails
from ....tools.online.wikipedia import Wikipedia

class TravelAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm, agent_process_queue,
                 agent_process_factory,
                 log_mode: str
        ):
        BaseAgent.__init__(self, agent_name, task_input, llm, agent_process_queue, agent_process_factory, log_mode)
        self.tool_list = {
            "hotel_location_search": HotelLocationSearch(),
            "hotel_search": HotelSearch(),
            "get_hotel_details": GetHotelDetails(),
            "airport_search": AirportSearch(),
            "flight_search": FlightSearch(),
            "restaurant_location_search": RestaurantLocationSearch(),
            "restaurant_search": RestaurantSearch(),
            "get_restaurant_details": GetRestaurantDetails(),
            "wikipedia": Wikipedia(),
        }
        self.tool_check_max_fail_times = 10
        self.tool_select_max_fail_times = 10
        self.tool_calling_max_fail_times = 10
        self.tools = self.config["tools"]
        self.workflow = self.config["workflow"]

    def load_flow(self):
        return

    def run(self):
        prompt = ""
        prefix = self.prefix
        task_input = self.task_input
        prompt += prefix
        request_waiting_times = []
        request_turnaround_times = []
        task_input = "The task you need to solve is: " + task_input
        prompt += task_input

        self.logger.log(f"{task_input}\n", level="info")

        rounds = 0

        # TODO replace fixed steps with dynamic steps
        for i, step in enumerate(self.workflow):
            prompt += f"\nIn step {rounds + 1}, you need to {step}. Output should focus on current step and don't be verbose!"
            self.logger.log(f"Step {i+1}: {step}\n", level="info")

            # Avoid calling tools in the last step
            tools_to_use = self.tools if i < len(self.workflow) - 1 else None

            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                message = Message(
                    prompt = prompt,
                    tools = tools_to_use
                )
            )

            response_message = response.response_message
            if i == 0:
                self.set_start_time(start_times[0])

            tool_calls = response.tool_calls

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            if tool_calls:
                self.logger.log(f"***** It starts to call external tools *****\n", level="info")

                function_responses = ""
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.tool_list[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    try:
                        function_response = function_to_call.run(function_args)
                        function_responses += function_response
                        prompt += function_response
                    except Exception:
                        continue

                self.logger.log(f"The solution to step {rounds+1}: It will call the {function_name} with the params as {function_args}. The tool response is {function_responses}\n", level="info")
            else:
                self.logger.log(f"The solution to step {rounds+1}: {response_message}\n", level="info")

            if i == len(self.workflow) - 1:
                self.logger.log(f"Final result is: {response_message}\n", level="info")
                final_result = response_message

            rounds += 1

        self.set_status("done")
        self.set_end_time(time=time.time())

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": rounds,
            "agent_waiting_time": self.start_time - self.created_time,
            "agent_turnaround_time": self.end_time - self.created_time,
            "request_waiting_times": request_waiting_times,
            "request_turnaround_times": request_turnaround_times,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TravelAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    args = parser.parse_args()
    agent = TravelAgent(args.agent_name, args.task_input)

    agent.run()
