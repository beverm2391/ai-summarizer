from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))

import openai

from package.utils import TokenUtil


# ! OPENAI CLASSES ---------------------------------------


class Completion:
    def __init__(self, temperature, max_tokens, stream=False, model="text-davinci-003", **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.model = model
        self.kwargs = kwargs

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, text):
        raw_response = openai.Completion.create(
            model=self.model,
            prompt=text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs
        )
        if self.stream:
            return raw_response
        elif len(raw_response['choices']) > 1:
            return [choice['text'].strip() for choice in raw_response['choices']]
        else:
            return raw_response['choices'][0]['text'].strip()


class CompletionV2:
    def __init__(self, temperature=0.7, max_tokens=1000, stream=False, model="text-davinci-003", **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.model = model
        self.kwargs = kwargs
        self.tokenutil = TokenUtil(model)

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, text):
        raw_response = openai.Completion.create(
            model=self.model,
            prompt=text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs
        )

        if self.stream:
            return raw_response
        elif len(raw_response['choices']) > 1:
            res_dicts = []
            for idx, choice in enumerate(raw_response['choices']):
                response = choice['text'].strip()
                tokens = raw_response['usage']['total_tokens']
                res_dict = {"response": response, "model": self.model, "temperature": self.temperature, "tokens": tokens}
                res_dicts.append(res_dict)
            return res_dicts
        else:
            response = raw_response['choices'][0]['text'].strip()
            tokens = raw_response['usage']['total_tokens']
            return {"response": response, "model": self.model, "temperature": self.temperature, "tokens": tokens}

class Chat:
    def __init__(self, temperature, system_message="You are a helpful assistant.", messages=None, model='gpt-3.5-turbo'):
        self.messages = []
        self.messages.append({"role": "system", "content": system_message})
        if messages is not None:
            self.messages += [{"role": "user", "content": message} for message in messages]
        self.model = model
        self.temperature = temperature

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, user_message: str):
        user_message = {"role": "user", "content": user_message}
        self.messages.append(user_message)
        raw_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )
        response_message = raw_response['choices'][0]['message']['content'].strip()
        self.messages.append(response_message)
        return response_message

class ChatV2:
    def __init__(self, temperature=0.7, system_message="You are a helpful assistant.", messages=None, model='gpt-3.5-turbo', max_tokens=2000):
        self.messages = []
        self.messages.append({"role": "system", "content": system_message})
        if messages is not None:
            self.messages += [{"role": "user", "content": message} for message in messages]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, user_message: str):
        user_message = {"role": "user", "content": user_message}
        self.messages.append(user_message)
        raw_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = raw_response['choices'][0]['message']['content'].strip()
        tokens = raw_response['usage']['total_tokens']
        res_message = {"role": "assistant", "content": text}
        self.messages.append(res_message)
        res_dict = {"response" : text, "messages": self.messages, "model": self.model, "temperature": self.temperature, "tokens": tokens}
        return res_dict