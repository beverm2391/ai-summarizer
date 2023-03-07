from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from operator import itemgetter
import numpy as np
import tiktoken

from typing import List, Dict, Tuple
import re
import time
import asyncio
from functools import wraps, partial
import openai

from dotenv import load_dotenv
import os

load_dotenv("../.env")

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
