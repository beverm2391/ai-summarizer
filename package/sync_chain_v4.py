from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")

sys.path.append(os.environ.get("PACKAGE_PATH"))

from package.utils import TokenUtil
from package.document import Document
from package.sync_api import ChatV2, CompletionV2


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

    def chat_summarize(self, instruction=None):
        responses = []
        # default_instruction = "Generate a lengthy and detailed list of direct quotes, main ideas, and passages by from the following text:\n\n"
        default_instruction = "Generate a lengthy, detailed summary from the following text:\n\n"
        for idx, chunk in enumerate(self.document.chunks):
            # init chat (we need to do this every time because the chat object aggregates messages)
            chat = ChatV2(model=self.model)
            # prepare prompt
            prompts = [default_instruction] + \
                ([instruction] if instruction else []) + [chunk]
            prompt = "\n\n".join(prompts)
            prompt_tokens = self.tokenutil.get_tokens(prompt)
            # debug
            print(
                f"Generating summary for chunk {idx+1} of {len(self.document.chunks)}: {chunk[:100]}...")
            print(f"Input token size {prompt_tokens}")
            assert prompt_tokens < self.token_limit - self.token_buffer, "Chunk too large"
            # get response
            data = chat("\n\n".join(prompts))
            # cleanup
            tokens = data["tokens"]
            response = data["response"]
            print(
                f"Summary {idx+1} generated, length {self.tokenutil.get_tokens(response)} tokens.\n\n{response}")
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

    def create_prompts(self, instruction, default_instruction, citation_instruction):
        prompts = []
        for chunk in self.document.chunks:
            prompt_parts = []

            if instruction:
                prompt_parts.append(instruction)
            else:
                prompt_parts.append(default_instruction)

            if self.document.citation:
                prompt_parts.append(citation_instruction)

            prompt_parts.append(chunk)
            prompt = "\n\n".join(prompt_parts)
            prompts.append(prompt)

            input_tokens = [self.tokenutil.get_tokens(
                prompt) for prompt in prompts]

        return prompts, input_tokens

    def completion_summarize(self, instruction=None):
        # setup inputs
        # default_instruction = f"Generate a list of main ideas and 2-3 notable quotes, from the following text, then write a summary including those main ideas. Make sure to include in text citations from this reference: {self.document.citation}. You do not need to include a reference list or bibliography:\n\n"
        default_instruction = f"""
            Follow these insructions:

            1. List the main ideas
            2. identify 2-3 direct quotes
            3. Identify potential weaknesses, assumptions and strengths of the author's argument in the context of considering how the article is relevant to social work practice, how the article contributes to a nuanced understanding of client behavior in the social environment, and how the author's argument might be connected to the profession's mandate to address issues of social justice with vulnerable populations.
            4.	Discuss how you might personally incorporate the point of the article into their own social work practice. For example, how might the content of the article be relevant to or applied to the student's practice in field placement? Or in future practice?
            """

        citation_instruction = f"Make sure to include in text citations from this reference: {self.document.citation if self.document.citation else ''}\n. You do not need to include a reference list or bibliography:\n\n"

        # prompts = ["\n\n".join([default_instruction] + ([f"OTHER INSTRUCTIONS: {instruction}"] if instruction else []) + [chunk]) for chunk in self.document.chunks]
        prompts, input_tokens = self.create_prompts(instruction, default_instruction, citation_instruction)
        
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

    def get_quotes(self, instruction=None):
        pass

    def sim_api_call(self, text, token_limit):
        print("Made sim api call")
        assert self.tokenutil.get_tokens(text) < token_limit, f"You passed {self.tokenutil.get_tokens(text)} tokens to the sim api call, which exceeds the limit of {token_limit}"
        # reduce the input to half. simulating a prompt and response
        return text[len(text)//2:]

    def aggregate(self, summaries, test=False):
        # ! check if we need to split the input
        assert summaries and len(summaries) > 0, "No summaries to aggregate"
        input_token_limit = 8192 - 2048 - 100  # 8192 is the limit, 2048 is the response, 100 is the buffer
        print(f"Input token limit: {input_token_limit} tokens.")
        joined = ' '.join(summaries)
        tokens = self.tokenutil.get_tokens(joined)

        # ! split the input if it is too large

        if tokens > input_token_limit:
            print(f"Aggregated input token size {tokens} exceeds limit {input_token_limit}.")
            print("Splitting input into multiple chunks...")
            chunks = self.tokenutil.split_tokens(joined, input_token_limit - 100)
            print(f"Split into {len(chunks)} chunks.")
            print(f"Chunk sizes: " + ', '.join([str(self.tokenutil.get_tokens(chunk)) for chunk in chunks]) + ".")
            print("Generating summaries for each chunk...")
            summaries = []
            for chunk in chunks:
                # make sure to pass test=True
                summaries.append(self.aggregate(chunk, test=test))
            print("Summaries generated.")
        else:
            print(f"Aggregated input token size {tokens} is within limit {input_token_limit}.")

        # ! if test, return the sim api call
        if test:
            print("Testing mode, using sim api call...")
            data = self.sim_api_call(' '.join(summaries), input_token_limit)
            summaries = [data]
            return

        # ! else base case
        prompt = f"""
                Aggregate the following information into a lengthy and detailed 1000-2000word paper with appropriate subheadings.
                Include a reference list with this citation: {self.document.citation}. 
                Inclde at least 2-3 direct quotes with the appropriate in-text citations.
                Make sure to output in markdown format:\n\n"
            """

        for summary in summaries:
            # print(f"Adding summary: {summary[:100]}...")
            prompt += f"{summary}\n\n"

        # print(f"Prompt:\n\n{prompt}")
        # aggregate the summaries
        print(f"Aggregating summaries...")
        system_prompt = "As a talented academic writer, you possess the exceptional ability to craft well-researched, coherent, and insightful papers. Your task now is to write a comprehensive 1000-2000 word essay on the topic provide it. Be sure to explore various aspects of the subject, and analyze the ways in which these elements have shaped our world. Remember to incorporate credible sources, provide a balanced perspective, and captivate your readers with your eloquent writing style. You will always use markdown format."
        
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
            assert isinstance(
                summary, str), f"Summary {idx + 1} is not a strings."

        final_summary_data = self.aggregate(self.summaries)
        final_summary = final_summary_data["response"].strip()
        print(f"Final summary:\n\n{final_summary}")
        print(f"Total tokens used: {self.total_tokens}")
        self.final_summary_data = final_summary_data
        return final_summary_data
    
    def __repr__(self):
        return f"Summarizer(document={self.document}), v4"
