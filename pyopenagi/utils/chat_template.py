class Query:
    def __init__(self,
            messages,
            tools = None
        ) -> None:
        self.messages = messages
        self.tools = tools

class Response:
    def __init__(
            self,
            response_message,
            tool_calls = None
        ) -> None:
        self.response_message = response_message
        self.tool_calls = tool_calls
