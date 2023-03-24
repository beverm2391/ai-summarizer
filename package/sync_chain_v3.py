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
        default_instruction = "Generate a lengthy, detailed summary from the following text:\n\n"
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
        default_instruction = "Generate a list of main ideas, from the following text, then write a summary including those main ideas:\n\n"
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

    def aggregate(self, summaries):
        assert len(summaries) == len(self.document.chunks), "Number of summaries must match number of chunks."
        prompt = "Aggregate the following summaries into a lengthy and detailed paper:\n\n"
        for summary in summaries:
            print(f"Adding summary: {summary[:100]}...")
            prompt += f"{summary}\n\n"

        # print(f"Prompt:\n\n{prompt}")
        # aggregate the summaries
        print(f"Aggregating summaries...")
        system_prompt = "As a talented academic writer, you possess the exceptional ability to craft well-researched, coherent, and insightful papers. Your task now is to write a comprehensive essay on a topic of your choice. Be sure to explore various aspects of the subject, and analyze the ways in which these elements have shaped our world. Remember to incorporate credible sources, provide a balanced perspective, and captivate your readers with your eloquent writing style."
        chat = ChatV2(model="gpt-4", temperature=0.7, system_message=system_prompt, max_tokens=2048)
        data = chat(prompt)
        print("Summary:\n\n" + data["response"])
        return data

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
        self.final_summary_data = final_summary_data
        return final_summary_data