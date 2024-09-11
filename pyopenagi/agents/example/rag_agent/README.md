# src/agents/agent_config

Each agent holds a config file in addition to the class specifying what to run. The agent config is a JSON file.

Each JSON file contains the following:

1. `name` : name of the agent
2. `description` : an array with one element containing the system prompt
3. `workflow` : an array with plaintext describing what the agent will do at each iteration. this is fed into the LLM running the agent
4. `tools` : an array with complex json objects
- `type` : type of tool, typically "function"
- `function` : if the type of function it contains data in the specific functions.

For more detailed information, cite each specific agent as an example and fit it for your purposes.
