from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from operator import itemgetter
import numpy as np
import tiktoken
from typing import List, Dict, Tuple
import os
import asyncio

from package.sync_api import Chat
from package.utils import *
from package.async_api_v2 import run_chat_async


def dot_product_similarity(doc_data: List[Dict], query_data: Dict) -> List[Tuple[int, float]]:
    query_embedding = query_data['embedding']
    doc_embeddings = [page['embedding'] for page in doc_data]
    tuples_list = [(page['page'], np.dot(query_embedding, embedding)) for page, embedding in zip(doc_data, doc_embeddings)]
    ordered_tuples = sorted(tuples_list, key=itemgetter(1), reverse=True)
    top_five_tuples = ordered_tuples[:5]
    return top_five_tuples


def embed_query(query: str):
    base_embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedding = base_embeddings.embed_query(query)
    return {"query": query, "embedding": embedding}


def embed_doc(text_list: List[str]):
    load_dotenv("../.env")
    api_key = os.getenv("OPENAI_API_KEY")
    base_embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"))
    doc_embeddings = base_embeddings.embed_documents(text_list)
    return doc_embeddings


def get_context(query_data, doc_data: List[Dict]) -> List[Dict]:
    top_five_tuples = dot_product_similarity(doc_data, query_data)
    context = []
    for item in top_five_tuples:
        page = item[0]
        data = {'page': page, 'similarity': item[1], 'text': doc_data[page - 1]
                ['content'], 'metadata': doc_data[page - 1]['metadata']}
        context.append(data)
    return context

def format_context(context: List[Dict], model: str, token_limit: int) -> str:
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


class Document:
    def __init__(self, fpath: str, model='gpt-3.5-turbo'):
        self.fpath = fpath
        self.model = model

    def generate_schemas(self, template, example):
        print("Generating schema...")
        if self.chunks is None:
            self.process_doc()
        prompts = [f"""
                Here is an example schema:
                {template[1]}

                Instructions:
                    1. I have given you a section from an article
                    2. Identify the main idea
                    3. Identify sub ideas of this section, if applicable, else leave blank
                    4. Use the example schema to generate a new schema with your ideas.
                    5. Do not include any other text, only output the schema and nothing else 

                Here is an example output:
                {example}

                Context:
                {chunk}

                Output:
                """
            for chunk in self.chunks]

        data = []
        asyncio.run(run_chat_async(prompts, data, model=self.model))
        schemas = [item["response"].replace("\n", " ").strip() for item in data]
        final_schema = [
            {
                "title": "Introduction",
                "prompt": "Write an introduction."
            },
            {*schemas},
            {
                "title": "Conclusion",
                "prompt": "Write a conclusion.",
            }
        ]
        self.schema = final_schema

    def pretty_print(self, data):
        return pretty_print(data)

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
        document = [{**page, "embedding": embedding}
                    for page, embedding in zip(document, doc_embeddings)]

        self.data = document
        self.page_text = ' '.join(
            [sanitize_text(page['content']) for page in document])
        self.metadata = [page['metadata'] for page in document]
        return self

    def get_chunks(self, chunk_size: int = 2800):
        tokenutil = TokenUtil(self.model)
        tokens = tokenutil.encode(self.page_text)
        # split into chunks of "chunk_size" tokens
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        # decode chunks
        decoded = [tokenutil.decode(chunk) for chunk in chunks]
        self.chunks = decoded
        return self

    def get_citation(self, format: str):
        citation_chat = Chat(temperature=0.9, model=self.model)
        get_citation_prompt = f"Use this metadata to generate a ciation in {format} format: \n\n{self.metadata}"
        final_citation = citation_chat(get_citation_prompt)
        self.citation = final_citation
        return self
