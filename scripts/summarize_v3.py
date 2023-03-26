from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))
from package.utils import write_file
from package.sync_chain_v6 import Chain
from package.document import Document

import time
import openai

OPENAI_API_KEY = os.environ.get('OPENAI-API-KEY')
openai.api_key = OPENAI_API_KEY


def main_process_doc(fpath):
    print("Starting doc processing...\n")
    start = time.perf_counter()
    doc = Document(fpath)
    doc.process_doc().get_chunks("gpt-3.5-turbo", 1000).get_citation('APA')
    elapsed = time.perf_counter() - start
    print(f"Doc processed in {elapsed:0.2f} seconds.")
    return doc


def main_generate_summaries(doc, instruction = None):
    print("Starting summary generation...\n")
    start = time.perf_counter()
    chain = Chain(doc, model="gpt-3.5-turbo")
    # completion_instruction = f"Generate a list of main ideas and 2-3 notable quotes, from the following text, then write a summary including those main ideas. Make sure to include in text citations from this reference: {doc.citation}. You do not need to include a reference list or bibliography:\n\n"
    chain.chat_summarize(instruction=instruction)
    elapsed = time.perf_counter() - start
    print(f"Summaries generated in {elapsed:0.2f} seconds.")
    return chain


def main():
    start = time.perf_counter()
    fpath = "data/structural-racism-black-women.pdf"
    base_filename = fpath.split('/')[-1].split('.')[0]

    doc = main_process_doc(fpath)

    instruction = """
            Follow these insructions:
            1. List the main ideas
            2. identify 2-3 direct quotes
            3. Identify potential weaknesses, assumptions and strengths of the author's argument in the context of considering how the article is relevant to social work practice, how the article contributes to a nuanced understanding of client behavior in the social environment, and how the author's argument might be connected to the profession's mandate to address issues of social justice with vulnerable populations.
            4. Discuss how you might personally incorporate the point of the article into their own social work practice.
            """

    chain = main_generate_summaries(doc, instruction=instruction)
    data, total_tokens = chain.aggregate(chain.summaries)


    new_filename = write_file(
        "data/outputs", f"{base_filename}_summary", "md", data["response"])

    elapsed = time.perf_counter() - start
    print(f"\nSummary saved to {new_filename}. Generated in {elapsed:0.2f} seconds. Total tokens: {total_tokens}.\n")



if __name__ == "__main__":
    main()