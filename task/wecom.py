import requests
import json
import os


class WeCom(object):
    def __init__(self, key: str):
        self.key = key

    def send(self, msg: str) -> None:
        headers = {"Content-Type": "application/json", "Charset": "UTF-8"}

        message = {"msgtype": "text", "text": {"content": msg}}

        response = requests.post(
            f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.key}",
            data=json.dumps(message),
            headers=headers,
        )

        if json.loads(response.text)["errmsg"] == "ok":
            print("WeCom send success")

        else:
            raise Exception(f"send msg to wecom error, err={json.loads(response.text)}")
