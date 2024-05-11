import os
import pytest
from src.tools.online.currency_converter import CurrencyConverterAPI
from dotenv import load_dotenv, find_dotenv

# Load environment variables once for all tests

# def test_currency_converter_api():
#     # Check if the necessary API key is available in the environment
#     load_dotenv(find_dotenv())
#     api_key = os.getenv("RAPID_API_KEY")
#     with pytest.raises(ValueError):
#         CurrencyConverterAPI()
#
#     # API key is present, proceed with creating an instance of the API
#     currency_converter_api = CurrencyConverterAPI()
#
#     # Test with incorrect parameter keys to expect a KeyError
#     with pytest.raises(KeyError):
#         bad_params = {
#             "from x": "USD",  # Incorrect key 'from x' should be 'from'
#             "to": "EUR",
#             "amount": 2
#         }
#         currency_converter_api.run(params=bad_params)
#
#     # Test with correct parameters
#     good_params = {
#         "from": "USD",
#         "to": "EUR",
#         "amount": 2
#     }
#     result = currency_converter_api.run(params=good_params)
#
#     # Check the type of the result, ensure it's a string as expected
#     assert isinstance(result, str), "The result should be a string"

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
