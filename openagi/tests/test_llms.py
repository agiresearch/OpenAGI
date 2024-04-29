import unittest
from openagi.src.agents.agent_process import AgentProcess

class TestAgentProcess(unittest.TestCase):

    def test_agent_creation(self):
        agent_process = AgentProcess(
            agent_name="Narrative Agent",
            prompt="Craft a tale about a valiant warrior on a quest to uncover priceless treasures hidden within a mystical island."
        )
        # Assuming AgentProcess has attributes that can be checked, for example, 'agent_name'
        self.assertEqual(agent_process.agent_name, "Narrative Agent", "Agent name does not match")
        # Add more assertions here as necessary to validate the properties of the agent_process object

if __name__ == "__main__":
    unittest.main()
