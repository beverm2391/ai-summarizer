from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")

sys.path.append(os.environ.get("PACKAGE_PATH"))

from package.sync_api import ChatV2, CompletionV2
from package.document import Document
from package.utils import TokenUtil


class Chain():
    def __init__(self, document: Document, model="gpt-4"):
        self.document = document
        self.tokenutil = TokenUtil(model)
        self.total_tokens = 0
        self.model = model
        self.summaries = []
        self.main_ideas = []
        self.token_limit = 8192
        self.token_buffer = 2000

    def chat_summarize(self, instruction = None):
        responses = []
        # default_instruction = "Generate a lengthy and detailed list of direct quotes, main ideas, and passages by from the following text:\n\n"
        default_instruction = "Generate a list of direct quotes, main ideas, and passages by from the following text:\n\n"
        for idx, chunk in enumerate(self.document.chunks):
            # init chat (we need to do this every time because the chat object aggregates messages)
            chat = ChatV2(model=self.model)
            # prepare prompt
            prompts = [default_instruction] + ([instruction] if instruction else []) + [chunk]
            prompt = "\n\n".join(prompts)
            prompt_tokens = self.tokenutil.get_tokens(prompt)
            # debug
            print(f"Generating summary for chunk {idx+1} of {len(self.document.chunks)}: {chunk[:100]}...")
            print(f"Input token size {prompt_tokens}")
            assert prompt_tokens < self.token_limit - self.token_buffer, "Chunk too large"
            # get response
            data = chat("\n\n".join(prompts))
            # cleanup
            tokens = data["tokens"]
            response = data["response"]
            print(f"Summary {idx+1} generated, length {self.tokenutil.get_tokens(response)} tokens.\n\n{response}")
            self.total_tokens += tokens
            responses.append(response)
            print("\n\n")
        self.summaries = responses

    def examine_data_structure(self, data):
        if isinstance(data, list):
            print(f"List of length {len(data)}")
            for item in data:
                self.examine_data_structure(item)
        elif isinstance(data, dict):
            print("Dict")
            for key, value in data.items():
                print(f"Key: {key}")
                self.examine_data_structure(value)
        else:
            print("Other")

    def completion_summarize(self, instruction = None):
        # setup inputs 
        default_instruction = "Generate a list of direct quotes, main ideas, and passages by from the following text:\n\n"
        prompts = ["\n\n".join([default_instruction] + ([instruction] if instruction else []) + [chunk]) for chunk in self.document.chunks]
        input_tokens = [self.tokenutil.get_tokens(prompt) for prompt in prompts]

        print(f"Input token sizes: {input_tokens}")
        print(f"Total input token size: {sum(input_tokens)}")
        print(f"Batch generating summaries for all chunks...")
        
        # get the summaries
        completion = CompletionV2(model="text-davinci-003", max_tokens=2000)
        responses = completion(prompts)

        # debug
        # self.examine_data_structure(responses)
        # print("\n\n")
        # print(responses)

        if isinstance(responses, list) and len(responses) > 1:
            self.summaries = [response["response"] for response in responses]
            tokens = sum([response["tokens"] for response in responses])
            print(f"Summaries generated, aggregate length {tokens} tokens.")
            self.total_tokens += tokens
        elif isinstance(responses, list) and len(responses) == 1:
            responses = responses[0]
        elif isinstance(responses, dict):
            self.summaries = [responses["response"]]
            tokens = responses["tokens"]
            print(f"Summary generated, length {tokens} tokens.")
            self.total_tokens += tokens
        
    def get_quotes(self, instruction = None):
        pass

    def aggregate(self, texts, token_limit=2000):
        print(f"Aggregating {len(texts)} texts...")

        # setup default instruction
        default_instruction = "Combine the following passages into one coherent essay:\n\n"
        default_instruction_tokens = self.tokenutil.get_tokens(default_instruction)

        # init chat
        chat = ChatV2(model=self.model, max_tokens=1800)

        # helper functions
        def aggregate_texts(texts):
            # re init because chat object aggregates messages which will go over the token limit
            chat = ChatV2(model=self.model, max_tokens=1800)
            prompt = default_instruction
            for text in texts:
                prompt += f"{text}\n\n"
            return chat(prompt)
        
        # case 1 (only one text)
        if len(texts) == 1:
            return texts[0]
        
        # case 2 (can fit all texts into one prompt)
        print("Checking if we can fit all texts into one prompt...")
        prompt_tokens = default_instruction_tokens + sum([self.tokenutil.get_tokens(text) for text in texts])
        if prompt_tokens < token_limit - self.token_buffer:
            print("Yes, we can!")
            return aggregate_texts(texts)
        
        # case 3 (can't fit all texts into one prompt)
        print("No, we can't. Splitting into chunks...")
        # for idx, text in enumerate(self.summaries):
        tokens = prompt_tokens
        all_text = '\n\n'.join(texts)

        char_interval = 4
        estimated_error = 200
        # estimate where to split the text
        # estimate (4 chars per token * total tokens we can fit in the prompt) - (estimated error to account for the char_interval not being exact)
        split_at = (char_interval*(token_limit - default_instruction_tokens)) - estimated_error*char_interval
        print(f"Splitting at {split_at} characters")
        # split the text
        chunks = [all_text[i:i+split_at] for i in range(0, len(all_text), split_at)]
        print(f"Split into {len(chunks)} chunks")
        # aggregate the chunks
        responses = []
        for idx, chunk in enumerate(chunks):
            print(f"Generating aggregation for chunk {idx+1} of {len(chunks)}: {chunk[:100]}...")
            response = aggregate_texts([chunk])
            tokens = response["tokens"]
            response = response["response"]
            print(f"Aggreate {idx+1} generated, length {tokens} tokens.\n\n{response}")
            self.total_tokens += tokens
            responses.append(response)
            print("\n\n")
        # pass the chunks recursively to this function
        return self.aggregate(responses)

    
    def chain_summarize(self, summary_type="completion", instruction=None):
        if self.summaries == []:
            if summary_type == "chat":
                self.chat_summarize(instruction)
            elif summary_type == "completion":
                self.completion_summarize(instruction)

        # debug
        print(f"Summaries:\n{self.summaries}")

        # tests
        assert self.summaries != [], "No summaries generated yet."
        assert isinstance(self.summaries, list), "Summaries must be a list."
        for idx, summary in enumerate(self.summaries):
            assert isinstance(summary, str), f"Summary {idx + 1} is not a strings."

        final_summary_data = self.aggregate(self.summaries)
        final_summary = final_summary_data["response"].strip()
        print(f"Final summary:\n\n{final_summary}")
        print(f"Total tokens used: {self.total_tokens}")
        return final_summary_data