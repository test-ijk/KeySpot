import os
import requests
from dotenv import load_dotenv


def notify(msg: str):
    """通过Bark发送通知

    Args:
        msg: 要发送的消息内容
    """
    try:
        # 使用python-dotenv加载.env.local文件
        load_dotenv("src/.env")

        # 获取BARK_URL环境变量
        bark_url = os.getenv("BARK_URL")
        if not bark_url:
            print("未在.env.local中找到BARK_URL配置,无法发送通知")
            return

        # 发送GET请求
        url = f"{bark_url}/{msg}"
        print(url)
        response = requests.get(f"{bark_url}{msg}")
        if response.status_code == 200:
            print("通知发送成功")
        else:
            print(f"通知发送失败,状态码:{response.status_code}")

    except Exception as e:
        print(f"发送通知时出错:{str(e)}")
