import os
import pytest
from src.tools.online.currency_converter import CurrencyConverterAPI
from dotenv import load_dotenv, find_dotenv
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

def main():
    test_currency_converter_api()

if __name__ == "__main__":
    main()
