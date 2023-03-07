import time

from mongodoc import Mongodoc
from async_api import run_chat_async
from sync_api import Chat, Completion

class AsyncChain:
    def __init__(self, mongodoc: Mongodoc):
        self.mongodoc = mongodoc
        # pass all the attributes from the mongodoc to the chain
        for attr in dir(mongodoc):
            if not callable(getattr(mongodoc, attr)) and not attr.startswith("__"):
                setattr(self, attr, getattr(mongodoc, attr))

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
        prompt = f"Combine the following passages:{' '.join([summary for summary in self.link_2_responses])} into an essay. Retain any in-text citations, don't add any new citations except for {self.citation} if you don't have it."
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