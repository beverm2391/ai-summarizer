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
from package.sync_chain_v2 import Chain

def main():
    print("Starting doc processing...\n")
    start = time.perf_counter()
    fpath = "/Users/beneverman/Documents/Coding/semantic-search/ai-summarizer/data/powers2017.pdf"
    doc = Document(fpath)
    doc.process_doc().get_chunks("gpt-3.5-turbo", 2000)
    elapsed = time.perf_counter() - start

    print(f"Doc processed in {elapsed:0.2f} seconds.")
    print("Starting chain...\n")

    start = time.perf_counter()
    chain = Chain(doc, model="gpt-3.5-turbo")
    chain.chain_summarize()
    elapsed = time.perf_counter() - start

    print(f"\n\nChain completed in {elapsed:0.2f} seconds.")

if __name__ == "__main__":
    main()

# ! This is set up for testing right now, to only run 1 chunk through the summarizer method
# ! remember to remove the [0] from the doc.chunks[0] in the chain_summarize method