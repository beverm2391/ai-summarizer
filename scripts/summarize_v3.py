from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))

OPENAI_API_KEY = os.environ.get('OPENAI-API-KEY')
import openai
openai.api_key = OPENAI_API_KEY

import time

from package.document import Document
from package.sync_chain_v3 import Chain


def main():
    # ! Process the document (load text)
    print("Starting doc processing...\n")
    start = time.perf_counter()

    fpath = "data/powers2017.pdf"
    doc = Document(fpath)
    doc.process_doc().get_chunks("gpt-3.5-turbo", 1000)

    elapsed = time.perf_counter() - start

    print(f"Doc processed in {elapsed:0.2f} seconds.")

    # ! Chunk the document, and generate summaries
    print("Starting summary generation...\n")

    start = time.perf_counter()
    chain = Chain(doc, model="gpt-3.5-turbo")
    chain.completion_summarize()
    elapsed = time.perf_counter() - start

    print(f"Summaries generated in {elapsed:0.2f} seconds.")

    # ! Aggregate the summaries

    data = chain.aggregate(chain.summaries)

    # ! Print final summary

    print(data['response'])


if __name__ == "__main__":
    main()