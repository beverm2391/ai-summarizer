from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from operator import itemgetter
import numpy as np
import tiktoken
from typing import List, Dict, Tuple
import os
import openai
from dotenv import load_dotenv

import sys
sys.path.append("/Users/beneverman/Documents/Coding/semantic-search/langchain-testing/package")
# package imports
from utils import *
from sync_api import Chat

load_dotenv(".env")


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

class Mongodoc:
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
        mongodoc = unpack(data)
        mongodoc = [{**page, "embedding": embedding} for page, embedding in zip(mongodoc, doc_embeddings)]

        self.data = mongodoc
        self.page_text = ' '.join([sanitize_text(page['content']) for page in mongodoc])
        self.metadata = [page['metadata'] for page in mongodoc]
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