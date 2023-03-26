from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")

sys.path.append(os.environ.get("PACKAGE_PATH"))

from package.utils import TokenUtil
from package.document import Document
from package.sync_api import ChatV2, CompletionV2
from package.async_api_v2 import run_chat_async

import asyncio

class Chain():
    def __init__(self, document: Document, model="gpt-4", test=False):
        self.document = document
        self.tokenutil = TokenUtil(model)
        self.total_tokens = 0
        self.model = model
        self.summaries = []
        self.main_ideas = []
        self.token_limit = 8192
        self.token_buffer = 2000
        self.test = test

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

    def chat_summarize(self, instruction=None):
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
        self.total_tokens += sum(input_tokens)

        print(f"Input token sizes: {input_tokens}")
        print(f"Total input token size: {sum(input_tokens)}")
        print(f"Batch generating summaries for all chunks...")

        data = []
        asyncio.run(run_chat_async(prompts, data, model="gpt-3.5-turbo"))

        self.summaries = [item["response"] for item in data]
        self.total_tokens = sum([self.tokenutil.get_tokens(summary) for summary in self.summaries])

    def get_quotes(self, instruction=None):
        pass

    def sim_api_call(self, text, token_limit):
        # print("Made sim api call")
        assert self.tokenutil.get_tokens(text) < token_limit, f"You passed {self.tokenutil.get_tokens(text)} tokens to the sim api call, which exceeds the limit of {token_limit}"
        # reduce the input to half. simulating a prompt and response
        return text[len(text)//2:]
    
    def get_chat_response(self, prompt, model):
        print(f"Aggregating summaries...")
        system_prompt = "As a talented academic writer, you possess the exceptional ability to craft well-researched, coherent, and insightful papers. Your task now is to write a comprehensive 1000-2000 word essay on the topic provide it. Be sure to explore various aspects of the subject, and analyze the ways in which these elements have shaped our world. Remember to incorporate credible sources, provide a balanced perspective, and captivate your readers with your eloquent writing style. You will always use markdown format."

        if model == 'gpt-4':
            max_response_tokens = 2048
        elif model == 'gpt-3.5-turbo':
            max_response_tokens = 1500
        else:
            max_response_tokens = 1500
        
        chat = ChatV2(model="gpt-4", temperature=0.7, system_message=system_prompt, max_tokens=max_response_tokens)
        data = chat(prompt)
        print("Final Summary:\n\n" + data["response"])
        self.total_tokens += data["tokens"]
        return data
    
    def build_combine_chunks_prompt(self, chunks):
        chunks = [chunks] if not isinstance(chunks, list) else chunks
        prompt = f"""
            Combine the following data to create coherent paragraphs.
            Make sure to include in text citations from this reference: {self.document.citation if self.document.citation else ''}.
            Include a reference list with as well.
            Make sure to output in markdown format:\n\n
        """
        for chunk in chunks:
            prompt += f"{chunk}\n\n"
        return prompt

    def build_final_summary_prompt(self, chunks):
        # handle accidental single string input
        chunks = [chunks] if not isinstance(chunks, list) else chunks
        # ! else base case
        prompt = f"""
                Aggregate the following information into a lengthy and detailed 1000-2000 word paper with appropriate subheadings.
                Include a reference list with this citation: {self.document.citation}. 
                Inclde at least 2-3 direct quotes with the appropriate in-text citations.
                Make sure to output in markdown format:\n\n"
            """
        for chunk in chunks:
            prompt += f"{chunk}\n\n"
        return prompt
    
        pass

    def aggregate(self, summaries, model='gpt-3.5-turbo'):
        # ! check if we need to split the input
        assert summaries and len(summaries) > 0, "No summaries to aggregate"
        #  ! total tokens - response tokens - prompt tokens = input token limit

        if model == 'gpt-4':
            max_input_tokens = 8192
            max_response_tokens = 2048
            max_prompt_tokens = 500
        elif model == 'gpt-3.5-turbo':
            max_input_tokens = 4096
            max_response_tokens = 1500
            max_prompt_tokens = 500
        else:
            max_input_tokens = 4096
            max_response_tokens = 1500
            max_prompt_tokens = 500
        
        input_token_limit = max_input_tokens - max_response_tokens - max_prompt_tokens
        print(f"Input token limit: {input_token_limit} tokens.")
        joined = ' '.join(summaries)
        tokens = self.tokenutil.get_tokens(joined)

        while tokens > input_token_limit:
            print(f"Aggregated input token size {tokens} exceeds limit {input_token_limit}.")
            print("Splitting input into multiple chunks...")
            chunks = self.tokenutil.split_tokens(joined, input_token_limit)
            print(f"Split into {len(chunks)} chunks. Processing them simultaneously...")
            data = []
            prompts = [self.build_combine_chunks_prompt(chunk) for chunk in chunks]
            self.total_tokens += sum([self.tokenutil.get_tokens(prompt) for prompt in prompts])
            # ! run the chat async
            asyncio.run(run_chat_async(prompts, data, max_tokens=max_response_tokens, model=model))
            print(f"THIS LENGTH SHOULD BE > 1: {len(data)}")
            for idx, summary in enumerate(data):
                print(f"Intermediate summary {idx+1}:\n\n{summary}")
            
            joined = ' '.join([summary["response"] for summary in data])
            total_output_tokens = self.tokenutil.get_tokens(joined)
            tokens = total_output_tokens
            self.total_tokens += total_output_tokens

        print(f"Time for the final summary...")
        prompt = self.build_final_summary_prompt(joined)
        data = self.get_chat_response(prompt, model)
        return data, self.total_tokens


    def aggregate_OLD(self, summaries):
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
                if self.test == True:
                    response = self.sim_api_call(chunk, input_token_limit)
                else:
                    print("Made it here")
                    assert self.tokenutil.get_tokens(chunk) < input_token_limit, f"Chunk size {self.tokenutil.get_tokens(chunk)} exceeds limit {input_token_limit}"
                    response = self.get_chat_response(chunk)['response']
                summaries.append(response)

            print("Summaries generated.")
            return self.aggregate(summaries)

        print(f"Aggregated input token size {tokens} is within limit {input_token_limit}.")
        if self.test:
            response_data = self.sim_api_call(joined, input_token_limit)
        else:
            prompt = self.build_summary_prompt(summaries)
            response_data = self.get_chat_response(prompt)
        return response_data
    
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
