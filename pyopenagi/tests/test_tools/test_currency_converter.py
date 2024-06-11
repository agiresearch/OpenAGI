import os
import pytest

from src.tools.online.currency_converter import CurrencyConverterAPI
from dotenv import load_dotenv, find_dotenv

@pytest.fixture(scope="module")
def test_rapid_api_key():
    load_dotenv(find_dotenv())
    if "RAPID_API_KEY" not in os.environ or not os.environ["RAPID_API_KEY"]:
        with pytest.raises(ValueError):
            currency_converter_api = CurrencyConverterAPI()
        pytest.skip("Rapid api key is not set.")
    else:
        return True

@pytest.mark.usefixtures("test_rapid_api_key")
def test_currency_converter_api():
    load_dotenv(find_dotenv())
    currency_converter_api = CurrencyConverterAPI()
    params = {
        "from": "USD",
        "to": "EUR",
        "amount": 2
    }
    result = currency_converter_api.run(params=params)
    print(result)
    assert isinstance(result, str)
