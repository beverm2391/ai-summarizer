import sys
sys.path.append("/Users/beneverman/Documents/Coding/semantic-search/langchain-testing/")
import asyncio

from package.mongodoc import Mongodoc
from package.async_chain import AsyncChain

def main():
    print("running main.py")
    fpath = "/Users/beneverman/Documents/Coding/semantic-search/langchain-testing/data/powers2017.pdf"
    doc = Mongodoc(fpath).process_doc().get_chunks(2800).get_citation("APA")

    async_chain = AsyncChain(doc)
    asyncio.run(async_chain.chain())
    async_chain.print_link_3()

    print("Complete")

if __name__ == "__main__":
    main()