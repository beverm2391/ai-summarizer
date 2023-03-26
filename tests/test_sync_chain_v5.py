from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))

from package.sync_chain_v5 import Chain

def test_aggregate():
    # simulate 10 chunks at 1000 tokens each
    test_doc_summaries = [
        ' '.join(["token" for i in range(1000)]) for i in range (10)
    ]

    # test the chain
    doc = Chain("placeholderfpath")
    doc.chunks = test_doc_summaries

    chain = Chain(doc, test=True)
    chain.summaries = chain.document.chunks
    chain.document.citation = ""

    chain.aggregate(chain.summaries)

    print("test_aggregate() passed")

if __name__ == "__main__":
    test_aggregate()