from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")

sys.path.append(os.environ.get("PACKAGE_PATH"))

from package.sync_api import Chat, Completion
from package.mongodoc import Mongodoc
import time


class Chain:
    def __init__(self, mongodoc: Mongodoc):
        self.mongodoc = mongodoc
        # pass all the attributes from the mongodoc to the chain
        for attr in dir(mongodoc):
            if not callable(getattr(mongodoc, attr)) and not attr.startswith("__"):
                setattr(self, attr, getattr(mongodoc, attr))

    def link_1(self):
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
        for idx, chunk in enumerate(self.chunks):
            print(f"Chunk {idx+1} of {len(self.chunks)}")
            page_responses = []
            for prompt in prompts:
                print(prompt["type"])
                chat = Chat(temperature=0.9, system_message=system_message)
                response = chat(f"CONTEXT:{chunk}\n\nQUERY:{prompt['prompt']}")
                page_responses.append(
                    {"Chunk": idx+1, "prompt_type": prompt["type"], "response": response})
            responses.append(page_responses)
        self.link_1_responses = responses

        print("Link 1 Complete")
        return self

    def print_link_1(self):
        for page in self.link_1_responses:
            for response in page:
                print(
                    f'Chunk: {response["Chunk"]}\nType: {response["prompt_type"]}\n\nResponse:\n{response["response"]}')

    def link_2(self):
        print("Starting Link 2:")
        llm = Completion(temperature=0.9, max_tokens=1000)
        summary_responses = []
        for page in self.link_1_responses:
            combine_prompt = f"Combine the following Main Ideas:\n{page[0]['response']}\n\nQuotes:\n{page[1]['response']}\n\nPassages:\n{page[2]['response']}\n\ninto a coherent writing. Retain any in-text citations, don't add any new citations except for {self.citation}\n\nSUMMARY:"
            summary_response = llm(combine_prompt)
            summary_responses.append(summary_response)
        self.link_2_responses = summary_responses
        print("Link 2 Complete")
        return self

    def print_link_2(self):
        for response in self.link_2_responses:
            print(response)

    def link_3(self):
        print("Starting Link 3:")
        llm = Completion(temperature=0.9, max_tokens=1000)
        prompt = f"combine the following passages:{' '.join([summary for summary in self.link_2_responses])} into an essay. Retain your in-text citations and make sure to include a reference list at the end of your essay using this citation: {self.citation}. Make sure you dont repeat anything."
        response = llm(prompt)
        self.link_3_response = response
        print("Link 3 Complete")
        return self

    def print_link_3(self):
        print(self.link_3_response)

    def link_4(self):
        with open("../data/apa_guidelines.txt", "r") as f:
            guidelines = f.read()

        print("Starting Link 4:")
        llm = Chat(temperature=0.9)
        guidelines = "https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/general_format.html"
        prompt = f"Essay:{self.link_3_response}\n\nFinalize thee essay based on the following APA guidelines: {guidelines}"
        response = llm(prompt)
        self.link_4_response = response
        print("Link 4 Complete")
        return self

    def print_link_4(self):
        print(self.link_4_response)

    def chain(self):
        start = time.perf_counter()
        print("Starting Chain")
        self.link_1().link_2().link_3().link_4()
        elapsed = time.perf_counter() - start
        print(f"Chain Complete in {elapsed:0.2f} seconds.")
        return self