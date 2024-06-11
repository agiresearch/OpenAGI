import os
import pytest

from src.tools.online.words_api import WordsAPI
from dotenv import load_dotenv, find_dotenv

@pytest.fixture(scope="module")
def test_rapid_api_key():
    load_dotenv(find_dotenv())
    if "RAPID_API_KEY" not in os.environ or not os.environ["RAPID_API_KEY"]:
        with pytest.raises(ValueError):
            words_api = WordsAPI()
        pytest.skip("Rapid api key is not set.")

@pytest.mark.usefixtures("test_rapid_api_key")
def test_words_api():
    words_api = WordsAPI()
    params = {
        "word": "look",
        "api_name": "typeOf",
    }
    result = words_api.run(params=params)
    print(result)
    assert isinstance(result, str)
