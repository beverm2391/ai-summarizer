>This is a copy of the article on my blog, [https://blocks.beneverman.com/projects/ai-summarizer].

# Accurate Summarization with AI

import Image from 'next/image'
import { Callout } from 'nextra-theme-docs'


<Callout type='info' emoji='👋'>
    Want to learn how to build stuff like this? DM me on [Twitter](https://twitter.com/beverm2391) and I'll see how I can help.
</Callout>

## Introduction
I started playing with [GPT-3]() in November of 2021. I quickly realized I could leverage [Natural Language Processing]() (NLP) to significantly speed up the pace of my schoolwork. Out of the box, [the newest Large Language Models]() (LLMs) can do quite a bit out of the box. They can:

1. Generate text
2. Converse
3. Answer open book questions (you provide the context in the prompt)
4. Answer closed book questions (semi-accurately)

They are essentially *complex data structures that can be queried with natural language*. When scaled up to billions of params, increased performance on un-trained benchmarks *emerges*, and we get these psuedo-intelligent feeling models like GPT-3. However, they are not perfect. For summarization, they have a few major limitations that I will discuss.

In my case, I want to generate accurate and thoughtful summaries (with sources) from a single or multiple source documents. Not just to read, but because summarizing scientific journals is one of my assignments! Later, I'll integrate this into my process of writing research papers.

## Problems with LLMs

<Callout type='warning'>
    Already familiar with LLMs? Skip to the [technical section](#code). The code is linked there.
</Callout>

**So where does native chatGPT/GPT-3.5 drop off in terms of performance?** (i.e. how far can we get before we run into problems?)

### Test 1: Summarize X (closed-book)

Let's do some tests. For this one, I'll ask chatGPT to summarize a concept for me. I'm going to follow the same theme throught this post, so you can compare the results of each method.

<Image src="/images/ai-summarizer/test-1.png" alt="test 1" className='image' width={1000} height={500} />

Not too bad! I love these results. I'm going to use this as a baseline for the rest of the tests. For most use cases, you could probably stop here - but I'm going to keep going.

### The Training Data Problem

It's important to understand where chatGPT is getting this information. It's not from the internet. It's from the training data - the collection of text that the model was trained on (think millions of passages). The problem is, the model has to be frozen in order to be used in a production environment. This means that it can't be updated with new information. So, if you want to use it to summarize a concept, you have to train it on that concept. 

The internet updates constantly, but the model doesn't - it has a cutoff date. 

<Image src="/images/ai-summarizer/what-year-is-it.png" alt="test 1" className='image' width={1000} height={500} />

Let's try another test. This time, I'll ask chatGPT to summarize a concept that it has never seen before.

<Image src="/images/ai-summarizer/future-events.png" alt="test 1" className='image' width={1000} height={500} />

This is much more subtle in our original test - the model is still able to generate a summary, but it's not up to date. It's not a problem in a lot of cases, but it's something to keep in mind. In my case, I need updated and accurate information about scientific journal articles that are published every day.

### The Hallucination Problem

So what happens when I ask a non-filtered model to summarize a concept that it has never seen before? Let's find out. I'm using text-davinci-003 (GPT-3.5) for this test.

<Image src="/images/ai-summarizer/sports-2023.png" alt="test 1" className='image' width={1000} height={500} />

That one's pretty easy to spot. What about this one?

<Image src="/images/ai-summarizer/political-events-2023.png" alt="test 1" className='image' width={1000} height={500} />

As OpenAI says, when LLM's **hallucinate** "you get plausible-sounding but incorrect answers."

In my use case, this is a major problem. I need accurate summaries that:
1. Are up to date
2. Are true!
3. Are not plagiarized (i.e. not copied from the internet)

## Test 2: Summarize X (open-book)
This time, I'll provide the context for chatGPT. This way it has all the information it needs to generate a summary, and won't hallucinate.

From the article, [*Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression: A Review*](https://pubmed.ncbi.nlm.nih.gov/29080589/).

The provided context:

<Image src="/images/ai-summarizer/summary-passage.png" alt="test 1" className='image' width={1000} height={500} />

The summary:

<Image src="/images/ai-summarizer/summary-response.png" alt="test 1" className='image' width={1000} height={500} />

Not bad! No hallucinations, and it's up to date. 

I need citations, though - so let's try that.

<Image src="/images/ai-summarizer/summary-citations.png" alt="test 1" className='image' width={1000} height={500} />

There it is, hallucinating again. I only provided context from one source (the article linked earlier), by Powers, Kleine, and Smits.Still this was a pretty good result.

### The Context Problem

Let's try the whole article. I'll provide the context for chatGPT. This way it has all the information it needs to generate a summary, and won't hallucinate.

<Image src="/images/ai-summarizer/token-limit.png" alt="test 1" className='image' width={1000} height={500} />

Problem. I can only provide a limited amount of context. I can't provide the entire article, because it's too long. This model, ChatGPT has a context limit of 4096 tokens, or about 3000 words (100 tokens ~= 75 words).

## Solutions

There are a few ways to get around these problems. I'll go over each one, while they get progressively more complex. 

### Method 1: Chunking

Probably the simplest method, just split the article into chunks, and provide the context for each chunk. This is a good method if you don't mind the summary being a little choppy.

Just one problem, what if there's an idea that spans multiple chunks? The model won't be able to connect the dots.

### Method 2: Semantic Search with Embeddings

This is a great method for finding the most relevant chunk to an idea. Essentially, you encode each chunk and the idea into a vector along with a query. Then, you find the most similar chunk to the query and use that as the context. Think of it as dynamic chunking.

### Method 3: Chaining

We can facilitate "reasoning" by chaining together several completions in sequence. This is a great method for generating a summary that flows well. This is how I'm going to do it.

1. Extract relevant information from chunks a, b, and c
2. Perform some intermediate reasoning
2. Generate a summary from the results of step 2

### Method 4: Chaining + Clustering + Topic Modeling

TBD - I'm still working on this one.

## Demo

Here's a demo of my summarizer in action. My plan is to integrate this into [semantics.cloud](https://www.semantics.cloud) (my semantic search app) within the next few weeks.

**Input:** [The Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression: A Review](https://pubmed.ncbi.nlm.nih.gov/29080589/)

**Output** (with citations, in 38.74 seconds):

>Cognitive-behavioral therapies (CBTs) serve as a beneficial treatment for symptoms of anxiety and depression. Research on CBT aims to identify the core mechanisms of change responsible for symptom improvement. There are two primary core mechanisms for CBTs for anxiety: fear extinction and threat reappraisal. Cognitive change is the most researched mechanism for CBTs for depression (Powers, 2017). Although these two theories may seem incompatible, they may represent differences in levels of analysis. Cognitive change is a critical factor in CBT for both anxiety and depression, regardless of how it is achieved.
>
>Powers' (2017) article, Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression, delves into the core mechanisms of cognitive behavioral therapy for treating mental health disorders like anxiety and depression. The article discusses various techniques such as fear extinction, threat reappraisal, exposure therapy, and cognitive restructuring, as well as newer techniques such as acceptance and mindfulness-based therapies, cognitive bias modification interventions, positive mood induction, exercise augmentation, and the importance of sudden gains and critical sessions. All these psychological processes are used to target faulty threat appraisals, which ultimately lead to symptom improvement and reduced anxiety. 
>
>Cognitive change plays a crucial role in the success of CBT for both anxiety and depression. In CBT for anxiety, fear extinction and threat reappraisal are central mechanisms of change. Fear extinction involves reducing the fear response to a conditioned stimulus by exposing individuals to the feared stimulus repeatedly without any adverse consequences. Threat reappraisal is a cognitive process that aims to change an individual's perception of a stressor, ultimately leading to decreased anxiety. In contrast, cognitive changes are primarily responsible for symptom improvement in CBTs for depression. Research has revealed cognitive restructuring as the most researched psychological mechanism in CBT for depression, observed in both antidepressant and CBT interventions.
>
>Additionally, various methods have been suggested to enhance threat reappraisal. Mindfulness-based approaches and acceptance and commitment therapy are two such methods. The study by Alfei, Ferrer Monti, Molina, et al., (2015) emphasizes that prediction error and trace dominance are essential factors in determining the fate of fear memories after post-training manipulations. On the other hand, Julian, Beard, Schmidt, et al., (2012) study presents how attention training can reduce attention bias and social stressor reactivity, while Vittengl, Clark, & Jarrett (2005) discuss the validity of sudden gains in acute-phase treatment of depression. 
>
>In conclusion, cognitive-behavioral therapies (CBTs) have proved effective in healing symptoms of anxiety and depression. Anxiety and depression have different core mechanisms for CBTs, but cognitive change is an integral factor in both. Fear extinction and threat reappraisal are two central mechanisms for CBT for anxiety, while cognitive restructuring is crucial for CBT for depression. Therapists can use various techniques, such as acceptance and mindfulness-based therapies, to enhance threat reappraisal. Therefore, understanding the core mechanisms of CBT encourages the development of future CBTs to improve mental health treatments. 
>
>References:
>
>Powers, M. B. (2017). Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression. Psychiatric Clinics of North America, 40, 611-623. doi:10.1016/j.psc.2017.08.010

Pretty sweet, right? I'm really happy with the results - it's a huge improvement over an out-of-the-box LLM.

Here's the debug output. You can see that in total, I made 9 calls in link one, 3 calls in link two (I'm batching them so you dont see them all in the output), and 1 call in link three. The total time was 38.74 seconds.

```
Starting Chain
Starting Link 1:
Response 5 of 9 complete.
Response time: 3.43 seconds.
Response 2 of 9 complete.
Response time: 3.96 seconds.
Response 1 of 9 complete.
Response time: 4.06 seconds.
Response 4 of 9 complete.
Response time: 5.53 seconds.
Response 8 of 9 complete.
Response time: 6.35 seconds.
Response 9 of 9 complete.
Response time: 6.35 seconds.
Response 7 of 9 complete.
Response time: 6.49 seconds.
Response 6 of 9 complete.
Response time: 8.76 seconds.
Response 3 of 9 complete.
Response time: 9.24 seconds.
Link 1 Complete
Link 1 Complete in 9.24 seconds.
Starting Link 2:
Link 2 Complete
Link 2 Complete in 8.27 seconds.
Starting Link 3:
Link 3 Complete
Link 3 Complete in 8.27 seconds.
Chain Complete in 38.74 seconds.
```

## Pipeline

I'm going to start with a diagram of the pipeline, and then go over each step in detail.

<Image src="/images/ai-summarizer/summarization-chain-v1.png" alt="pipeline" className='image' width={700} height={500} />

<span id='code'/>
## Code Breakdown

<Callout>
    All of the code for this project is available on my Github in [this repo](https://www.github.com/beverm2391/ai-summarizer).
</Callout>

<Callout type='warning'>
    This is going to be a bit lengthy and technical. If you're not interested in the code, you can skip to the [next section](#conclusion).
</Callout>

We're going to start with some setup. Here are the dependencies to install, and the imports we'll need.

### Dependencies 

```
python-dotenv
langchain
openai
tiktoken
```

### Imports
``` python copy
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
```

Here I'm just retrieving the OpenAI API key from my `.env` file and setting it as the API key for the OpenAI SDK. 

### Config
``` python copy
from dotenv import load_dotenv
import os
load_dotenv("../.env")
OPENAI_API_KEY = os.environ.get('OPENAI-API-KEY')
import openai
openai.api_key = OPENAI_API_KEY
```

Okay! Time for some real code! First - utility functions.

### Utils

``` python copy
import re

# this prevents the metadata from being None which causes errors
def sanitize_metadata(data):
    for item in data:
        meta = item.metadata
        for key, value in meta.items():
            if value is None:
                meta[key] = ""
    return data
```

This first function is called `sanitize_metadata` and takes in a list of items and checks each item's metadata for any values that are `None`. If it finds any `None` values, it replaces them with an empty string. The function then returns the original list with the updated metadata. We'll use this to sanatize the PDF's metadata. If we don't do this, we can hit errors when we try to pass the metadata to the OpenAI API.

``` python copy
def sanitize_text(text):
    # Replace any non-alphanumeric character with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace any multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text
```


This function is called `sanitize_text` and it takes in a string and performs three operations on it. First, it replaces any non-alphanumeric character (i.e., any character that is not a letter or a number) with a space using the `re.sub()` function from the `re` module. Next, it replaces any multiple spaces with a single space using the same `re.sub()` function. Finally, it removes any leading or trailing whitespace using the `strip()` method. This function is used to sanitize the text of the PDF's pages. Just like the metadata, if we don't do this, we can hit errors when we try to pass the text to the OpenAI API.

``` python copy
def unpack (data):
    return [{'page' : idx + 1, 'content' : page.page_content, 'metadata' : page.metadata} for idx, page in enumerate(data)]
```

This function, `unpack` is a helper function that takes in the data returned from the `langchain` PDFLoader module and returns a list of dictionaries.

Okay, that's it for the utils. Let's move on to more interesting stuff.

### Sync API

These next two classes are synchronous and interact with the OpenAI API. They're used to generate the completions and embeddings.

``` python copy
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

``` 

The `Completion` class is a wrapper for the OpenAI API function `openai.Completion.create()`. It takes in four parameters: `temperature`, `max_tokens`, `stream`, and `model`, as well as any additional keyword arguments. 

The `temperature` parameter controls the creativity of the generated text. Higher values result in more varied, surprising responses, while lower values result in more predictable, conservative responses.

The `max_tokens` parameter controls the length of the generated response, measured in the number of tokens.

The `stream` parameter, when set to `True`, returns a generator that yields multiple responses. When set to `False`, the method will return a single response.

The `model` parameter specifies which OpenAI language model to use.

The `__call__` method takes a string `text` (or a list of strings) as input, and generates and returns a text completion (or a list of completions) using the specified OpenAI API parameters.

``` python copy
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
```

The `Chat` class is a wrapper for the OpenAI API function `openai.ChatCompletion.create()`. It takes in three parameters: `temperature`, `system_message`, and `messages`.

The `temperature` parameter controls the creativity of the generated text, as in the `Completion` class.

The `system_message` parameter is a string that represents the first message that the chatbot will send to the user.

The `messages` parameter is a list of strings representing messages that have already been exchanged between the user and the chatbot.

The `model` parameter specifies which OpenAI language model to use.

The `__call__` method takes a string `user_message` as input, which represents the message sent by the user. The method then adds the user's message to the `messages` list, and generates a response using the OpenAI API function `openai.ChatCompletion.create()`. The method returns the response message as a string and adds it to the `messages` list.

There's one problem with this class, though. The `openai.ChatCompletion.create()` function is synchronous and can only handle one request at a time. This means that if we want to use this class to generate multiple chatbot responses, we have to wait for each response to be generated before we can generate the next one. Referring back to the data flow diagram, each of those 4 boxes (citations, passages, quotes, and main ideas) would have to wait for the previous one to finish before it could start. 

This is not good. Imagine if a burger cook had to wait for the previous burger to finish cooking before they could start the next one. No one would get fed. In our case, since we're going to make a bunch of requests to the OpenAI API, we need to make sure that we can make all of those requests at the same time (asynchronously). Unfortunately, this is going to take some finesse - but it will be worth it.

### Async API

Ths is definitely the most complicated part of the code (and the code I'm most proud of). Watching these asynchronous functions work is so satisfying. To give some perspective, it might take 5-6 seconds to generate a single chatbot response using the synchronous API. But using the asynchronous API, we can generate theoretically infinite chatbot responses in that same amount of time (it's I/O bound). I say theoretically because there's a limit to how many requests you can make to the OpenAI API in a given amount of time. But we won't be anywhere near that limit.

``` python copy
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run
```

The `async_wrap` function is a decorator that takes a function as input and returns a new function that can be executed asynchronously using asyncio. 

``` python copy

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
    response_message = raw_response['choices'][0]['message']['content'].strip()
    return response_message

```
The `chat_response` function uses the OpenAI API to generate a response to a given message. It takes in the temperature, model, and message as arguments, and returns the generated response message. I didn't use the `Chat` class here because I wanted to be able to use the `async_wrap` decorator on this function (it might have worked, but I didn't want to risk it).

``` python copy
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
```

The `asyncChatResponse` function uses the `async_wrap` decorator to wrap the `chat_response` function and execute it asynchronously using asyncio. It takes the same arguments as `chat_response`, as well as two additional arguments: `response_list` and `messages_list`. It generates a response to the given message, appends it to the `response_list`, and prints information about the response time and progress. 

The `run_chat_async` function is the main function that runs the chatbot asynchronously. It takes in a list of messages and two empty lists (`messages_list` and `response_list`) as arguments. It creates an asynchronous task for each message in the `messages_list`, using the `asyncChatResponse` function to generate a response to each message. These responses are appended to the `response_list`. The function uses `asyncio.gather()` to run all the tasks concurrently and wait for them to complete. Once all the tasks are finished, the `response_list` will contain all the generated responses, in the order they were generated.

### Document Class

This class is used to work with the PDF document. We'll start with the helper functions.

``` python copy
def dot_product_similarity(doc_data: List[Dict], query_data: Dict) -> List[Tuple[int, float]]:
    query_embedding = query_data['embedding']
    doc_embeddings = [page['embedding'] for page in doc_data]
    tuples_list = [(page['page'], np.dot(query_embedding, embedding)) for page, embedding in zip(doc_data, doc_embeddings)]
    ordered_tuples = sorted(tuples_list, key=itemgetter(1), reverse=True)
    top_five_tuples = ordered_tuples[:5]
    return top_five_tuples

def embed_query(query: str):
    base_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedding = base_embeddings.embed_query(query)
    return {"query" : query, "embedding" : embedding}

def embed_doc(text_list : List[str]):
    load_dotenv("../.env")
    api_key = os.getenv("OPENAI_API_KEY")
    base_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    doc_embeddings = base_embeddings.embed_documents(text_list)
    return doc_embeddings

def get_context(query_data, doc_data: List[Dict]) -> List[Dict]:
    top_five_tuples = dot_product_similarity(doc_data, query_data)
    context = []
    for item in top_five_tuples:
        page = item[0]
        data = {'page': page, 'similarity' : item[1], 'text': doc_data[page - 1]['content'], 'metadata' : doc_data[page - 1]['metadata']}
        context.append(data)
    return context

def get_tokens(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens
```

`dot_product_similarity` is a function that takes a list of dictionaries `doc_data`, where each dictionary represents a page and contains the page's embedding, and a dictionary `query_data` representing a query and containing the query's embedding. The function computes the dot product similarity between the query embedding and each page embedding in `doc_data`. It then returns the top five pages with the highest similarity score, where each page is represented by its page number and its similarity score.

`embed_query` is a function that takes a query as a string and returns the query's embedding as a dictionary containing both the query and its embedding.

`embed_doc` is a function that takes a list of strings `text_list` and returns a list of embeddings, where each embedding corresponds to one of the strings in `text_list`.

`get_context` takes a query and a list of dictionaries `doc_data`, calculates the dot product similarity between the query and each page embedding in `doc_data`, and returns the top five pages with the highest similarity scores. For each of these top five pages, the function creates a dictionary containing the page number, its similarity score, the page's text content, and the page's metadata.

`get_tokens` is a function that takes a string `string` and a model as a string `model` and returns the number of tokens in the string. It uses the `encoding_for_model` function from the `tiktoken` library to encode the string with the specified model, and then returns the length of the resulting encoding as the number of tokens in the string.

``` python copy
def format_context(context: List[Dict], model: str, token_limit : int) -> str:
    """Returns a string of the first 1024 tokens of the context."""
    context_string = ""
    meta_list = []
    encoding = tiktoken.encoding_for_model(model)
    for idx, item in enumerate(context):
        sanitized_text = item['text'].replace("\n", " ")
        context_string += f"Page: {item['page']}\n\nText: {sanitized_text}\n\n"
        meta_list.append(item['metadata'])
        tokens = get_tokens(context_string, model)
        if tokens > token_limit:
            encoded_text = encoding.encode(context_string)
            # cut it down to the token limit
            encoded_text = encoded_text[:token_limit]
            # decode it back to a string
            context_string = encoding.decode(encoded_text)
            # some testing to make sure it worked
            tokens = get_tokens(context_string, model)
            assert tokens <= token_limit, f"format context function failed to cut context down far enough. tokens: {tokens}"
            break
    return context_string, meta_list
```

`format_context` is a function that takes a list of dictionaries `context`, a model as a string `model`, and a token limit as an integer `token_limit`. The function creates a string representation of the context by concatenating the text content of each page in `context` along with its page number. It also creates a list of metadata for each page in `context`.

The function then uses the `get_tokens` function to count the number of tokens in the context string, encoded using the specified model. If the number of tokens exceeds the specified token limit, the function reduces the size of the context string to the token limit by encoding the string, truncating the encoding to the token limit, and then decoding it back to a string. The function also performs a test to ensure that the resulting string is no longer than the specified token limit.

The function then returns the formatted context string and the list of metadata for each page in `context`.

Okay, that's it for helper functions. Now we can get to the main class, `Document`. The `Document` class represents a PDF document and provides methods for processing the document, splitting it into chunks, and generating a citation for it.

``` python copy
class Document:
    def __init__(self, fpath : str):
        self.fpath = fpath

    def process_doc(self):
        loader = PyMuPDFLoader(self.fpath)
        # load the data
        unsanitized = loader.load()
        # make sure the metadata is not None
        data = sanitize_metadata(unsanitized)
        # get the doc embeddings
        doc_embeddings = embed_doc([page.page_content for page in data])
        # unpack the data and add the embeddings
        document = unpack(data)
        document = [{**page, "embedding": embedding} for page, embedding in zip(document, doc_embeddings)]

        self.data = document
        self.page_text = ' '.join([sanitize_text(page['content']) for page in document])
        self.metadata = [page['metadata'] for page in document]
        return self
    
    def get_chunks(self, chunk_size : int):
        enc = tiktoken.encoding_for_model("text-davinci-003")
        tokens = enc.encode(self.page_text)
        # split into chunks of 2800 tokens
        chunks = [tokens[i:i+2800] for i in range(0, len(tokens), 2800)]
        # decode chunks
        decoded = [enc.decode(chunk) for chunk in chunks]
        self.chunks = decoded
        return self
    
    def get_citation(self, format: str):
        citation_chat = Chat(temperature=0.9)
        get_citation_prompt = f"Use this metadata to generate a ciation in {format} format: \n\n{self.metadata}"
        final_citation = citation_chat(get_citation_prompt)
        self.citation = final_citation
        return self
```

The `__init__` method takes a file path as a string `fpath` and initializes the `Document` object with this file path.

The `process_doc` method loads the document using the `PyMuPDFLoader` class, sanitizes the metadata using the `sanitize_metadata` function, and obtains embeddings for each page using the `embed_doc` function. The method then unpacks the data and adds the embeddings to each page in the document. Finally, the method sets the `data`, `page_text`, and `metadata` attributes of the `Document` object and returns the object.

The `get_chunks` method splits the `page_text` attribute of the `Document` object into chunks of a specified size, encoded using the `text-davinci-003` model. The method sets the `chunks` attribute of the `Document` object and returns the object.

The `get_citation` method generates a citation for the `Document` object using the `Chat` class. The method prompts the user to use the metadata to generate a citation in a specified format, and then uses the `Chat` class to generate the citation. The method sets the `citation` attribute of the `Document` object and returns the object.

### Async Chain

Time to put it all together into one large chain. I'm going to show the whole class here, but I'll break it down into smaller pieces below.

``` python copy
class AsyncChain:
    def __init__(self, document: Document):
        self.document = document
        # pass all the attributes from the document to the chain
        for attr in dir(document):
            if not callable(getattr(document, attr)) and not attr.startswith("__"):
                setattr(self, attr, getattr(document, attr))

    async def link_1(self):
        print("Starting Link 1:")
        citation_qualifier = f"Use this citation: {self.citation} to cite your work."
        main_ideas_prompt = f"Identify and list 2-3 main ideas from the context. {citation_qualifier}"
        quotes_prompt = f"Identify and list 2-3 relevant quotes from the context. {citation_qualifier}"
        passages_prompt = f"Identify and list 2-3 relevant passages from the context. {citation_qualifier}"

        prompts = [
            {"type": "main_ideas", "prompt": main_ideas_prompt},
            {"type": "quotes", "prompt": quotes_prompt},
            {"type": "passages", "prompt": passages_prompt}
        ]

        system_message = "You are a helpful assistant that is very good at problem solving who thinks step by step. You always cite direct quotes and paraphrases with the appropriate in-text citation."

        responses = []
        hardcode_prompts = [[f"CONTEXT:{chunk}\n\nQUERY:{prompt['prompt']}" for prompt in prompts]
                            for idx, chunk in enumerate(self.chunks)]
        flattened_prompts = [
            item for sublist in hardcode_prompts for item in sublist]

        await run_chat_async(flattened_prompts, responses)

        # split the responses into chunks of 3
        response_chunks = []
        n = len(prompts)
        for i in range(0, len(responses), n):
            response_chunk = responses[i:i+n]
            response_chunks.append(response_chunk)

        # create a list of dicts
        response_dicts = []
        for idx, chunk in enumerate(response_chunks):
            page_dict = []
            for prompt, response in zip(prompts, chunk):
                page_dict.append(
                    {"Chunk": idx+1, "prompt_type": prompt["type"], "response": response})
            response_dicts.append(page_dict)
        self.link_1_responses = responses
        # format [[{chunk: 1, prompt_type: main_ideas, response: ...}, ...], ...]
        self.link_1_response_dicts = response_dicts
        print("Link 1 Complete")
        return self

    def print_link_1(self):
        for page in self.link_1_response_dicts:
            for response in page:
                print(f'Chunk: {response["Chunk"]}\nType: {response["prompt_type"]}\n\nResponse:\n{response["response"]}\n')

    async def link_2(self):
        print("Starting Link 2:")
        prompts = []
        for page in self.link_1_response_dicts:
            combine_prompt = f"Combine the following Main Ideas:\n{page[0]['response']}\n\nQuotes:\n{page[1]['response']}\n\nPassages:\n{page[2]['response']}\n\ninto a coherent writing. Retain any in-text citations, don't add any new citations except for {self.citation}\n\nSUMMARY:"
            prompts.append(combine_prompt)
        llm = Completion(temperature=0.9, max_tokens=1000)
        responses = llm(prompts)
        self.link_2_responses = responses
        print("Link 2 Complete")
        return self
    
    def print_link_2(self):
        for response in self.link_2_responses:
            print(response)

    def link_3(self):
        print("Starting Link 3:")
        chat = Chat(temperature=0.9)
        prompt = f"Combine the following passages:{' '.join([summary for summary in self.link_2_responses])} into an essay. Retain your in-text citations and make sure to include a reference list at the end of your essay using this citation: {self.citation}."
        response = chat(prompt)
        self.link_3_response = response
        print("Link 3 Complete")
        return self
    
    def print_link_3(self):
        print(self.link_3_response)

    async def chain(self):
        overall_start = time.perf_counter()
        start = time.perf_counter()
        print("Starting Chain")
        await self.link_1()
        elapsed = time.perf_counter() - start
        print(f"Link 1 Complete in {elapsed:0.2f} seconds.")
        start = time.perf_counter()
        await self.link_2()
        elapsed = time.perf_counter() - start
        print(f"Link 2 Complete in {elapsed:0.2f} seconds.")
        start = time.perf_counter()
        self.link_3()
        print(f"Link 3 Complete in {elapsed:0.2f} seconds.")
        elapsed = time.perf_counter() - start
        overall_elapsed = time.perf_counter() - overall_start
        print(f"Chain Complete in {overall_elapsed:0.2f} seconds.")
        return self
```

This code defines a class called `AsyncChain` that contains methods for running a chain of tasks asynchronously using asyncio. 

The `__init__` method takes a Document object as input and passes all of its attributes to the `AcyncChain` object. 

The `link_1` method generates prompts for identifying main ideas, relevant quotes, and relevant passages from the context. It then uses the `run_chat_async` function to generate responses to these prompts for each chunk of text in the document. It splits the responses into chunks of 3 and creates a list of dictionaries containing the chunk number, prompt type, and response for each response. It sets the `link_1_responses` attribute to the list of responses and `link_1_response_dicts` attribute to the list of dictionaries. 

The `print_link_1` method prints out the responses for each chunk in a formatted way. 

The `link_2` method generates prompts for combining the main ideas, quotes, and passages from each chunk into a coherent writing. It then uses the completion class to generate responses to these prompts. It sets the `link_2_responses` attribute to the list of responses. 

The `print_link_2` method prints out the responses for each prompt in a simple way. 

The `link_3` method generates a prompt for combining the responses from `link_2` into a coherent essay, using the Chat class from earlier to generate the final response. It sets the `link_3_response` attribute to the generated response. 

The `print_link_3` method prints out the final essay response. 

The `chain` method runs the entire chain of tasks asynchronously. It executes `link_1`, `link_2`, and `link_3` in order, waiting for each one to complete before executing the next. The elapsed time for each link is printed, as well as the overall time for the entire chain. At the end, the method returns the AsyncChain object.

### Running the Chain with Main

``` python copy
def main():
    print("running main.py")
    fpath = "/data/scientific-journal-1.pdf"
    doc = Document(fpath).process_doc().get_chunks(2800).get_citation("APA")

    async_chain = AsyncChain(doc)
    asyncio.run(async_chain.chain())
    async_chain.print_link_3()

    print("Complete")

if __name__ == "__main__":
    main()
```

And there you have it - a fully functional asynchronous chain of tasks that can be run on a single document to generate a coherent summary (with citations) in about 30 seconds. I'm proud of this one. 

<span id='conclusion'/>
## Conclusion

So what's next? You didn't think I was going to stop there, did you?

### Improving the Chain

I also want to be able to summarize groups of documents at once, to speed up the process of reading and synthesizing the literature. And what about actually writing new ideas, instead of just summarizing? My end goal is to build a tool that I can integrate into my own research workflow. I want to be able to express my ideas in natural language and have AI articulate them into a rough draft in real time. 

This is all going to require some improvement to the chain.

### Using K-Means Clustering to Handle Groups of Documents

As I scale this up to handle groups of documents, I'll need to use some clustering algorithm to group similar ideas together across documents. I'll probably use K-Means clustering, which is a simple and effective algorithm for grouping data points into clusters.

### Using Topic Modeling to Improve 

I'm also going to need to improve the quality of the summaries. I'm not sure how I'm going to do this yet, but I'm thinking about using topic modeling to identify the main topics in each chunk of text and then using those topics to generate more relevant prompts. I'm going to play around with [BERTTopic](https://maartengr.github.io/BERTopic/index.html). I think this might lead to a significant performance in crease, especially when it comes to integrating the chain into my own research workflow.

## Final Thoughts

I hope you enjoyed this post. I'm really excited about the potential of this project. I'm going to keep working on it and I'll post updates as I go. Feel free to reach out to me on [Twitter](https://twitter.com/beneverman) or by [email](mailto:ben@beneverman.com) if you have any questions or comments. Cheers!