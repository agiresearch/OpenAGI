from src.agents.agent_process import AgentProcess
import pytest

def test_agent_creation():
    agent_process = AgentProcess(
        agent_name="Narrative Agent",
        message="Craft a tale about a valiant warrior on a quest to uncover priceless treasures hidden within a mystical island."
    )
    # Use plain assert statements for testing conditions
    assert agent_process.agent_name == "Narrative Agent", "Agent name does not match"
    # Add more assertions here as necessary to validate the properties of the agent_process object
