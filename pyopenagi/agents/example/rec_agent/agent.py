
from ...react_agent import ReactAgent

import numpy as np
class RecAgent(ReactAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm,
                 agent_process_queue,
                 agent_process_factory,
                 log_mode
        ):
        ReactAgent.__init__(self, agent_name, task_input, llm, agent_process_queue, agent_process_factory, log_mode)

    def automatic_workflow(self):
        return super().automatic_workflow()

    def manual_workflow(self):
        workflow = [
            {
                "message": "identify the tool that you need to call to obtain information.",
                "tool_use": ["imdb_top_movies", "imdb_top_series"]
            },
            {
                "message": "based on the information, give recommendations for the user based on the constrains. ",
                "tool_use": None
            }
        ]
        return workflow

    def run(self):
        return super().run()
