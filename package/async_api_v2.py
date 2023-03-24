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

def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run

# ! Chat --------------------------------------------------------------------------------------------

def chat_response(temperature, model, message):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ]
    raw_response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    text = raw_response['choices'][0]['message']['content'].strip()
    tokens = raw_response['usage']['total_tokens']
    res_message = {"role": "assistant", "content": text}
    messages.append(res_message)
    res_dict = {"response" : text, "messages": messages, "model": model, "temperature": temperature, "tokens": tokens}
    return res_dict

async_chat_response = async_wrap(chat_response)

async def asyncChatResponse(temperature, model, message, response_list, messages_list):
    start_time = time.perf_counter()
    response = await async_chat_response(temperature, model, message)
    elapsed = time.perf_counter() - start_time

    index = messages_list.index(message) + 1
    length = len(messages_list)
    print(f"Response {index} of {length} complete.")
    print(f"Response time: {elapsed:0.2f} seconds.")
    response_list.append(response)

async def run_chat_async(messages_list, response_list, temperature=0.7, model='gpt-3.5-turbo'):
    await asyncio.gather(*(asyncChatResponse(temperature, model, message, response_list, messages_list) for message in messages_list))