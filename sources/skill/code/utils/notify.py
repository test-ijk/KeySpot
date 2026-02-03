import os
import requests
from dotenv import load_dotenv


def notify(msg: str):

    try:
        load_dotenv("src/.env")

        bark_url = os.getenv("BARK_URL")
        if not bark_url:
            print("none")
            return

        response = requests.get(f"{bark_url}{msg}?level=critical&volume=3&badge=1")
        if response.status_code == 200:
            print("success")
        else:
            print(f"fail:{response.status_code}")

    except Exception as e:
