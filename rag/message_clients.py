import logging
from typing import Optional
import requests
import json
import time

from datetime import datetime

RECEIVE_PROMPT_ENDPOINT = '/receive-prompt'
EMPTY_QUEUE_INTERVAL_SEC = 60
MESSAGE_FAILURE_SLEEP_SEC = 15


class MessageData:
    def __init__(self, id: str, input: str, date: datetime, output: str):
        self.id = id
        self.input = input
        self.date = date
        self.output = output

    @property
    def json(self):
        return json.dumps({'id': self.id, 'input': self.input, 'date': self.date, 'output': self.output})

    def __str__(self):
        return str(self.json)


class MessageClient:
    def __init__(self):
        pass

    async def messages(self):
        yield None


class APIClient(MessageClient):
    def __init__(self, base_url: str, token: Optional[str]):
        super().__init__()
        self.base_url = base_url
        self.token = token

    def next_message(self, endpoint):
        url = self.base_url + endpoint
        if self.token is not None:
            headers = {'Authorization': 'Bearer ' + self.token}
        else:
            headers = {}
        response = requests.get(url, headers=headers)
        return response

    # TODO do this endpoint in the API
    def send_message(self, endpoint, message):
        url = self.base_url + endpoint
        if self.token is not None:
            headers = {'Authorization': 'Bearer ' + self.token}
        else:
            headers = {}
        response = requests.post(url, headers=headers, json=message)
        return response

    def messages(self) -> MessageData:
        while True:
            response = self.next_message(endpoint=RECEIVE_PROMPT_ENDPOINT)
            if response.status_code == 204:
                logging.info(f"Currently no jobs to process, sleeping for {EMPTY_QUEUE_INTERVAL_SEC} seconds...")
                time.sleep(EMPTY_QUEUE_INTERVAL_SEC)
                continue
            elif response.ok:
                message = response.json()
                message_data = MessageData(
                    id=message['_id'],
                    input=message['input'],
                    date=message['date'],
                    output=message['output']
                )
                yield message_data
            else:
                logging.warning(f"Received response code {response.status_code} from APIMessaging call. "
                                f"Backing off for {MESSAGE_FAILURE_SLEEP_SEC} seconds.")
                time.sleep(MESSAGE_FAILURE_SLEEP_SEC)
