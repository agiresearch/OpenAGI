import os
import pytest
from dotenv import load_dotenv, find_dotenv
from openagi.src.tools.online.currency_converter import CurrencyConverterAPI
from openagi.src.tools.online.wolfram_alpha import WolframAlpha
def test_currency_converter_api():
    load_dotenv(find_dotenv())
    if "RAPID_API_KEY" not in os.environ or not os.environ["RAPID_API_KEY"]:
        with pytest.raises(ValueError):
            currency_converter_api = CurrencyConverterAPI()
    else:
        currency_converter_api = CurrencyConverterAPI()
        with pytest.raises(KeyError):
            params = {
                "from x": "USD",
                "to": "EUR",
                "amount": 2
            }
            result = currency_converter_api.run(params=params)
            print(result)

        params = {
            "from": "USD",
            "to": "EUR",
            "amount": 2
        }
        result = currency_converter_api.run(params=params)
        print(result)
        assert isinstance(result, str)

def test_wolfram_alpha():
    load_dotenv(find_dotenv())
    if "WOLFRAM_ALPHA_APPID" not in os.environ or not os.environ["WOLFRAM_ALPHA_APPID"]:
        with pytest.raises(ValueError):
            wolfram_alpha = WolframAlpha()
    else:
        wolfram_alpha = WolframAlpha()

        query = "What is the square root of 1764?"
        result = wolfram_alpha.run(query)
        print(result)
        assert isinstance(result, str)

def main():
    test_currency_converter_api()
    test_wolfram_alpha()

if __name__ == "__main__":
    main()
